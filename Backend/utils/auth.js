import jwt from "jsonwebtoken"

const httpCookies = (res, user) => {
    const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET, { expiresIn: "1h" })

    const httpCookie = {
        httpOnly: true,
        secure: false,
        sameSite: "strict",
        maxAge: 60 * 60 * 1000
    }

    res.cookie("token", token, httpCookie)
}
export default httpCookies