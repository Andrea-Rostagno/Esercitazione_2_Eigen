#include <iostream>
#include"Eigen/Eigen"
#include<iomanip>

using namespace Eigen;
using namespace std;

VectorXd qr_solution(MatrixXd A,VectorXd b){
    HouseholderQR<Matrix2d>qr(A);
    MatrixXd Q= qr.householderQ();
    MatrixXd R= qr.matrixQR().triangularView<Upper>();
    MatrixXd R_inv=R.inverse();
    MatrixXd Q_trans=Q.transpose();
    VectorXd sol=R_inv*(Q_trans*b);

    return sol;
}

VectorXd lu_solution(MatrixXd A,VectorXd b){
    PartialPivLU<Matrix2d>lu(A);
    MatrixXd P=lu.permutationP();
    MatrixXd L=lu.matrixLU().triangularView<UnitLower>();
    MatrixXd U=lu.matrixLU().triangularView<Upper>();
    MatrixXd y=L.inverse()*P*b;
    MatrixXd x=U.inverse()*y;

    return x;
}

int main()
{
    setlocale(LC_ALL,"C");

    //sistema 1:
    Matrix2d A;
    A<<5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01,-9.992887623566787e-01;
    Vector2d b;
    b<<-5.169911863249772e-01, 1.672384680188350e-01;
    Vector2d sol_esatta;
    sol_esatta<<-1.0,-1.0;

    //metodo QR
    VectorXd sol=qr_solution(A,b);
    cout<<setprecision(16)<<"la soluzione qr del sistema 1 "<<(char)232<<":"<<endl<<sol<<endl;
    double errore_relativo =(sol-sol_esatta).norm() / sol_esatta.norm();
    cout<<setprecision(16)<<"L'errore relativo qr del sistema 1 "<<(char)232<<": "<<errore_relativo<<endl;

    //metodo LU
    VectorXd x=lu_solution(A,b);
    cout<<setprecision(16)<<"la soluzione lu del sistema 1 "<<(char)232<<":"<<endl<<x<<endl;
    double errore_relativo1 =(x-sol_esatta).norm() / sol_esatta.norm();
    cout<<setprecision(16)<<"L'errore relativo lu del sistema 1 "<<(char)232<<": "<<errore_relativo1<<endl;


    //sistema 2:
    A<<5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01,-8.324762492991313e-01;
    b<<-6.394645785530173e-04, 4.259549612877223e-04;
    sol_esatta<<-1.0,-1.0;

    //metodo QR
    sol=qr_solution(A,b);
    cout<<setprecision(16)<<"la soluzione qr del sistema 2 "<<(char)232<<":"<<endl<<sol<<endl;
    errore_relativo =(sol-sol_esatta).norm() / sol_esatta.norm();
    cout<<setprecision(16)<<"L'errore relativo qr del sistema 2 "<<(char)232<<": "<<errore_relativo<<endl;

    //metodo LU
    x=lu_solution(A,b);
    cout<<setprecision(16)<<"la soluzione lu del sistema 2 "<<(char)232<<":"<<endl<<x<<endl;
    errore_relativo1 =(x-sol_esatta).norm() / sol_esatta.norm();
    cout<<setprecision(16)<<"L'errore relativo lu del sistema 2 "<<(char)232<<": "<<errore_relativo1<<endl;


    //sistema 3:
    A<<5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01,-8.320502947645361e-01;
    b<<-6.400391328043042e-10, 4.266924591433963e-10;
    sol_esatta<<-1.0,-1.0;

    //metodo QR
    sol=qr_solution(A,b);
    cout<<setprecision(16)<<"la soluzione qr del sistema 3 "<<(char)232<<":"<<endl<<sol<<endl;
    errore_relativo =(sol-sol_esatta).norm() / sol_esatta.norm();
    cout<<setprecision(16)<<"L'errore relativo qr del sistema 3 "<<(char)232<<": "<<errore_relativo<<endl;

    //metodo LU
    x=lu_solution(A,b);
    cout<<setprecision(16)<<"la soluzione lu del sistema 3 "<<(char)232<<":"<<endl<<x<<endl;
    errore_relativo1 =(x-sol_esatta).norm() / sol_esatta.norm();
    cout<<setprecision(16)<<"L'errore relativo lu del sistema 3 "<<(char)232<<": "<<errore_relativo1<<endl;


    return 0;
}

