import jwt from "jsonwebtoken"

const authMiddleware = (req, res, next) => {
    try {
        const token = req.cookies.token;
        if (!token) {
            return res.status(401).json({
                success: false,
                message: "Token not found in cookies."
            })
        }
        const tokenData = jwt.verify(token, process.env.JWT_SECRET, (err) => {
            if (err) {
                return res.status(401).json({
                    success: false,
                    message: "Unauthorized"
                })
            }
            req.body.userId = tokenData.userId;
            next();
        })

        if (!tokenData) {
            return res.status(401).json({
                success: false,
                message: "Unauthorized"
            })
        }
        req.body.userId = tokenData.userId;
        next();
    } catch (error) {
        return res.status(500).json({
            success: false,
            message: "Internal Server Error"
        })
    }
}

export default authMiddleware
