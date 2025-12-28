from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.spatial import ConvexHull
from scipy.optimize import fsolve
import numpy as np
from scipy import constants as con
import json

eV2J = 1.602e-19
k_b = con.k / eV2J  # eV*K-1

class Phase_script:
    def __init__(
        self,
        file,
        Tem,
        degreeMax=50,
        x0=0,
        x1=0.25,
        cg=-0.04626020292721478 * 8,
        nn=4000,
        x_density=1000,
        common_tangent_k=0,
        yMin=0,
        degree=None,
    ):
        self.degreeMax = degreeMax
        self.degree = degree
        self.x0 = x0
        self.x1 = x1
        self.file = file
        self.Tem = Tem
        self.beta = 1 / (k_b * self.Tem)  # eV-1
        self.cg = cg
        self.nn = nn
        self.x_density = x_density
        self.common_tangent_k = common_tangent_k
        self.yMin = yMin
        self.readAndFit()

    def readAndFit(self):
        self.x, self.y = self.readG()

        if self.degree:
            rmse, coef = self.polyFitG(self.degree)
            self.degreeMin = self.degree
            self.coefficient = coef[::-1]
            self.rmse = rmse
        else:
            rmses, coefs = [], []
            for degree in range(4, self.degreeMax):
                rmse, coef = self.polyFitG(degree)
                rmses.append(rmse)
                coefs.append(coef)
            self.degreeMin = np.argmin(rmses)
            self.coefficient = coefs[self.degreeMin][::-1]
            self.rmses = rmses

        self.coefficient[-1] = -self.yMin
        # self.coefficient[-2] = 0
        self.x_pred = np.linspace(self.x0, self.x1, 10000)
        self.y_pred = np.poly1d(self.coefficient)
        self.y_deriv = np.polyder(self.y_pred)
        self.y_deriv_2 = np.polyder(self.y_deriv)
        """
        self.spinodal_point = self.calcSpinodalPoint()
        self.tangent_point, self.y_tan, self.xk, self.yk = self.calcCommonTangent()
        """

    def readG(self):
        """
        Read G, y2 means subtract 2 * G_phase2
        :param file: MetaDynamics json file
        :param Tem: temperature in Kelvin
        :return: x, y2
        """
        with open(self.file, "r") as f:
            data = json.load(f)

        keys = ["bias_pot", "betaG"]

        x = np.array(data[keys[1]]["x"])
        y = np.array(data[keys[1]]["y"]) / self.nn
        y = y - y[0]
        y2 = y + x * self.cg * self.beta + self.common_tangent_k * x + self.yMin
        return (
            x[int(self.x0 * self.x_density) : 1 + int(self.x1 * self.x_density)],
            -y2[int(self.x0 * self.x_density) : 1 + int(self.x1 * self.x_density)],
        )

    def polyFitG(self, degree):
        """
        :param degree:
        :return: rmse, coef
        """
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(self.x.reshape(-1, 1))
        model = LinearRegression()
        model.fit(X_poly, self.y.reshape(-1, 1))

        # Root Mean Squared Error
        y_pred = model.predict(X_poly)
        mse = mean_squared_error(self.y, y_pred)
        rmse = np.sqrt(mse)
        # print(f"degree: {degree}, Root mean Squared Error: {rmse}")
        return rmse, model.coef_[0]

    def calcSpinodalPoint(self, guess):
        """
        calc spinodal point by f‘’=0
        :param coefficient:
        :return: spinodal_point
        """
        spinodal_point = fsolve(self.y_deriv_2, guess)
        return spinodal_point
        # return (min(spinodal_point, key=lambda _: abs(_ - self.guess[1])), min(spinodal_point, key=lambda _: abs(_ - self.guess[2])))


    def calcCommonTangent(self, guess):
        def eq(uv):
            # Define the equations for common tangent
            u, v = uv
            return [
                self.y_deriv(u) - self.y_deriv(v),
                self.y_pred(u) + (v - u) * self.y_deriv(u) - self.y_pred(v),
            ]

        uv = fsolve(eq, guess)
        return uv


    def resetY(self, uv):
        """
        Resets the antiderivative function from the common tangent.
        :param uv: Common tangent point of x, can get by calling self.calcCommonTangent2(guess)
        """
        UV = self.y_deriv(uv)
        self.y_pred -= np.poly1d([UV[0], 0])
        d = self.y_pred(uv)[0]
        self.y_pred -= d
        self.y_deriv -= UV[0]

        dy = np.poly1d([UV[0], d])
        self.y -= dy(self.x)

