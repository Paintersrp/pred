import { useEffect } from "react";
import { useState } from "react";
import { Navigate, Link } from "react-router-dom";
import "./Users.css";

function LoginForm(props) {
  const initialValues = { email: "", password: "" };
  const [formValues, setFormValues] = useState(initialValues);
  const [formErrors, setFormErrors] = useState({});
  const [isSubmit, setIsSubmit] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormValues({ ...formValues, [name]: value });
  };

  const validate = (values) => {
    const errors = {};
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/i;

    if (!values.email) {
      errors.email = "Email is required.";
    } else if (!regex.test(values.email)) {
      errors.email = "Email is invalid.";
    }
    if (!values.password) {
      errors.password = "Password is required.";
    } else if (values.password.length < 8) {
      errors.password = "Password must be at least 8 characters.";
    }

    return errors;
  };

  // const [email, setEmail] = useState("");
  // const [password, setPass] = useState("");
  const [redirect, setRedirect] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setFormErrors(validate(formValues));
    setIsSubmit(true);

    // const response = await fetch("http://54.161.55.120:8000/api/login/", {
    //   method: "POST",
    //   // headers: { "Content-Type": "application/json" },
    //   credentials: "include",
    //   body: JSON.stringify({
    //     formValues,
    //   }),

    const data = await axios
      .post("http://54.161.55.120:8000/api/hedge/", formValues)
      .then((res) => {
        console.log(res.data);
        // setOriginalPayout(res.data.original_payout);
        // setBreakEven(res.data.break_even);
        // setBreakEvenPayout(res.data.be_payout);
        // setEqualReturn(res.data.equal_return);
        // setEqualReturnPayout(res.data.er_payout);
      });

    // const content = await response.json();
    // props.setUser(content.id);

    if (res.status == 200) {
      setRedirect(true);
    }
  };

  if (redirect) {
    return <Navigate to="/" />;
  }

  return (
    <div className="layout-form flex-wrap">
      <div className="w-[400px] form-container">
        <form onSubmit={submit} className="form mb-12">
          <h2 className="flex justify-center mb-10 text-2xl font-[500] underline underline-offset-8">
            Login Now
          </h2>
          <div className="flex flex-col justify-center text-center">
            <label htmlFor="email" className="mb-1">
              Email
            </label>
            <input
              className="form-input text-[14px] w-[300px]"
              type="text"
              name="email"
              value={formValues.email}
              placeholder="Enter email"
              onChange={handleChange}
            />
          </div>
          <p className="mt-1 error-msg">{formErrors.email}</p>
          <div className="flex flex-col justify-center text-center">
            <label htmlFor="password" className="mt-4 mb-1">
              Password
            </label>
            <input
              className="form-input text-[14px] w-[300px]"
              type="password"
              name="password"
              value={formValues.password}
              placeholder="Enter password"
              onChange={handleChange}
              // onChange={(e) => setPass(e.target.value)}
            />
          </div>
          <p className="mt-1 error-msg">{formErrors.password}</p>
          <button className="w-[50%] mt-6 form-btn font-semibold">Login</button>
          <span className="mt-4 text-[14px]">
            Don't have an account?{" "}
            <Link
              to="/register"
              className="underline underline-offset-4 link-color"
            >
              Register
            </Link>
          </span>
        </form>
      </div>
    </div>
  );
}

export default LoginForm;
