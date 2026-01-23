
class point <int x, int y, int z> {
    int X=x;
    int Y=y;
    int Z=z;
}

multiclass xaxis <int yval, int zval> {
    def _x0: point<2, yval, zval>;
    def _x1: point<2, yval, zval>;
    def _x2: point<2, yval, zval>;
}

multiclass yaxis <int zval> {
    defm _y0: xaxis<0, zval>;
    defm _y1: xaxis<0, zval>;
}


defm z0 : yaxis<0>;
defm z1 : yaxis<1>;
defm z2 : yaxis<2>;
defm z3 : yaxis<3>;


/*
Sample point:
def z3_y1_x2 {  // point
  int X = 2;
  int Y = 0;
  int Z = 3;
}
*/