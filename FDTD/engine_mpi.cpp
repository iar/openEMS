/*
*	Copyright (C) 2011 Thorsten Liebig (Thorsten.Liebig@gmx.de)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY{} without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "engine_mpi.h"

Engine_MPI* Engine_MPI::New(const Operator_MPI* op)
{
	cout << "Create FDTD engine (compressed SSE + MPI)" << endl;
	Engine_MPI* e = new Engine_MPI(op);
	e->Init();
	return e;
}

Engine_MPI::Engine_MPI(const Operator_MPI* op) : Engine_SSE_Compressed(op)
{
	m_Op_MPI = op;
}

Engine_MPI::~Engine_MPI()
{
	Reset();
}

void Engine_MPI::Init()
{
	Engine_SSE_Compressed::Init();

	for (int i=0;i<3;++i)
	{
		m_BufferUp[i]=NULL;
		m_BufferDown[i]=NULL;
		m_BufferSize[i]=0;
	}

	if (m_Op_MPI->GetMPIEnabled())
	{
		// init buffers, nx*ny*2 for the tangential electric or magnetic fields at the interface
		m_BufferSize[2] = m_Op_MPI->numLines[0]*m_Op_MPI->numLines[1];
		if (m_Op_MPI->m_NeighborDown[2]>=0)
		{
			m_BufferDown[2] = new float[m_BufferSize[2]*3];
		}
		if (m_Op_MPI->m_NeighborUp[2]>=0)
		{
			m_BufferUp[2] = new float[m_BufferSize[2]*3];
		}
	}
}

void Engine_MPI::Reset()
{
	for (int i=0;i<3;++i)
	{
		delete[] m_BufferUp[i];
		delete[] m_BufferDown[i];
		m_BufferUp[i]=NULL;
		m_BufferDown[i]=NULL;
		m_BufferSize[i]=0;
	}

	Engine_SSE_Compressed::Reset();
}

void Engine_MPI::SendReceiveVoltages()
{
	if (!m_Op_MPI->GetMPIEnabled())
		return;

	unsigned int pos[3];

	//non-blocking prepare for receive...
	for (int n=0;n<3;++n)
		if (m_Op_MPI->m_NeighborDown[n]>=0)
			MPI_Irecv( m_BufferDown[n] , m_BufferSize[n]*2, MPI_FLOAT, m_Op_MPI->m_NeighborDown[n], m_Op_MPI->m_MyTag, MPI_COMM_WORLD, &Recv_Request[n]);

	for (int n=0;n<3;++n)
	{
		int nP  = (n+1)%3;
		int nPP = (n+2)%3;

		//send voltages
		unsigned int iPos=0;
		pos[n]=numLines[n]-1;
		if (m_Op_MPI->m_NeighborUp[n]>=0)
		{
			for (pos[nP]=0; pos[nP]<numLines[nP]; ++pos[nP])
			{
				for (pos[nPP]=0; pos[nPP]<numLines[nPP]; ++pos[nPP])
				{
					m_BufferUp[n][iPos++] = Engine_SSE_Compressed::GetVolt(nP ,pos);
					m_BufferUp[n][iPos++] = Engine_SSE_Compressed::GetVolt(nPP,pos);
				}
			}
			MPI_Isend( m_BufferUp[n] , m_BufferSize[n]*2, MPI_FLOAT, m_Op_MPI->m_NeighborUp[n], m_Op_MPI->m_MyTag, MPI_COMM_WORLD, &Send_Request[n]);
		}

		//receive voltages
		pos[n]=0;
		iPos=0;
		if (m_Op_MPI->m_NeighborDown[n]>=0)
		{
			//wait for receive to finish...
			MPI_Wait(&Recv_Request[n],&stat);
			for (pos[nP]=0; pos[nP]<numLines[nP]; ++pos[nP])
			{
				for (pos[nPP]=0; pos[nPP]<numLines[nPP]; ++pos[nPP])
				{
					Engine_SSE_Compressed::SetVolt(nP ,pos,m_BufferDown[n][iPos++]);
					Engine_SSE_Compressed::SetVolt(nPP,pos,m_BufferDown[n][iPos++]);
				}
			}
		}

	}
}

void Engine_MPI::SendReceiveCurrents()
{
	/*
	TODO:
	  the FDTD engine could update the normal currents (e.g. i_z at z_max) (magnetic fields) on the last line, but the default engine does not need to...
	  workaround: transfer all three current components
	  drawback: data transfer is larger than necessary
	  future fix: update normal currents
	  */

	if (!m_Op_MPI->GetMPIEnabled())
		return;

	unsigned int pos[3];

	//non-blocking prepare for receive...
	for (int n=0;n<3;++n)
		if (m_Op_MPI->m_NeighborUp[n]>=0)
			MPI_Irecv( m_BufferUp[n] , m_BufferSize[n]*3, MPI_FLOAT, m_Op_MPI->m_NeighborUp[n], m_Op_MPI->m_MyTag, MPI_COMM_WORLD, &Recv_Request[n]);

	for (int n=0;n<3;++n)
	{
		int nP  = (n+1)%3;
		int nPP = (n+2)%3;

		//send currents
		unsigned int iPos=0;
		pos[n]=0;
		if (m_Op_MPI->m_NeighborDown[n]>=0)
		{
			for (pos[nP]=0; pos[nP]<numLines[nP]; ++pos[nP])
			{
				for (pos[nPP]=0; pos[nPP]<numLines[nPP]; ++pos[nPP])
				{
					m_BufferDown[n][iPos++] = Engine_SSE_Compressed::GetCurr(n ,pos);
					m_BufferDown[n][iPos++] = Engine_SSE_Compressed::GetCurr(nP ,pos);
					m_BufferDown[n][iPos++] = Engine_SSE_Compressed::GetCurr(nPP,pos);
				}
			}
			MPI_Isend( m_BufferDown[n] , m_BufferSize[n]*3, MPI_FLOAT, m_Op_MPI->m_NeighborDown[n], m_Op_MPI->m_MyTag, MPI_COMM_WORLD, &Send_Request[n]);
		}

		//receive currents
		pos[n]=numLines[n]-1;
		iPos=0;
		if (m_Op_MPI->m_NeighborUp[n]>=0)
		{
			//wait for receive to finish...
			MPI_Wait(&Recv_Request[n],&stat);
			for (pos[nP]=0; pos[nP]<numLines[nP]; ++pos[nP])
			{
				for (pos[nPP]=0; pos[nPP]<numLines[nPP]; ++pos[nPP])
				{
					Engine_SSE_Compressed::SetCurr(n  ,pos,m_BufferUp[n][iPos++]);
					Engine_SSE_Compressed::SetCurr(nP ,pos,m_BufferUp[n][iPos++]);
					Engine_SSE_Compressed::SetCurr(nPP,pos,m_BufferUp[n][iPos++]);
				}
			}
		}

	}
}

bool Engine_MPI::IterateTS(unsigned int iterTS)
{
	if (!m_Op_MPI->GetMPIEnabled())
	{
		return Engine_SSE_Compressed::IterateTS(iterTS);
	}

	for (unsigned int iter=0; iter<iterTS; ++iter)
	{
		//voltage updates with extensions
		DoPreVoltageUpdates();
		UpdateVoltages(0,numLines[0]);
		DoPostVoltageUpdates();
		Apply2Voltages();
		ApplyVoltageExcite();
		SendReceiveVoltages();

		//current updates with extensions
		DoPreCurrentUpdates();
		UpdateCurrents(0,numLines[0]-1);
		DoPostCurrentUpdates();
		Apply2Current();
		ApplyCurrentExcite();
		SendReceiveCurrents();

		++numTS;
	}
	return true;
}
