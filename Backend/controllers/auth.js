import userModel from "../models/user.js"
import bcrypt from "bcrypt"
import validate from "validator"
import httpCookies from "../utils/auth.js"


const register = async (req, res) => {
    try {
        const { name, email, password } = req.body;

        if (!name || !email || !password) {
            return res.status(400).json({ success: false, message: "All fields are required" });
        }

        if (!validate.isEmail(email)) {
            return res.status(400).json({
                success: false, message: "Invalid Email."
            })
        }

        const isEmailExist = await userModel.findOne({ email })
        if (isEmailExist) {
            return res.status(400).json({ success: false, message: "Email Aldready Exist." })
        }

        const salt = await bcrypt.genSalt(10)
        const hashPassword = await bcrypt.hash(password, salt)


        const newUser = new userModel({ name, email, password: hashPassword });
        const user = await newUser.save();

        res.status(201).json({ success: true, data: user });
        httpCookies(res, user);

    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
}

const login = async (req, res) => {
    try {
        const { email, password } = req.body;
        if (!email || !password) {
            return res.status(400).json({ success: false, message: "All fields are required" })
        }

        const user = await userModel.findOne({ email })
        if (!user) {
            return res.status(400).json({ success: false, message: "User Not Found." })
        }

        const isPasswordMatch = await bcrypt.compare(password, user.password)

        if (!isPasswordMatch) {
            return res.status(400).json({ success: false, message: "Invalid Password." })
        }

        httpCookies(res, user);
        res.status(200).json({ success: true, data: user })

    } catch (error) {
        res.status(500).json({ success: false, error: error.message })
    }
}

export { register, login }