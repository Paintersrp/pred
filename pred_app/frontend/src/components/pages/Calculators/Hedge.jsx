import { useState } from "react";
import axios from "axios";
import { Link } from "react-router-dom";
import "./Calculators.css";

function Hedge() {
  const [toggleState, setToggleState] = useState(1);
  const [typeState, setTypeState] = useState(1);

  const [oddsNum, setOdds] = useState();
  const [wagerNum, setWager] = useState();
  const [hedgeNum, setHedge] = useState();

  const [originalPayout, setOriginalPayout] = useState(0);
  const [breakEven, setBreakEven] = useState(0);
  const [breakEvenPayout, setBreakEvenPayout] = useState(0);
  const [equalReturn, setEqualReturn] = useState(0);
  const [equalReturnPayout, setEqualReturnPayout] = useState(0);

  const toggleOption = (index) => {
    setToggleState(index);
  };

  const toggleType = (index) => {
    setTypeState(index);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const calcNums = { wagerNum, oddsNum, hedgeNum, typeState };

    const data = await axios
      .post("http://127.0.0.1:8000/api/hedge/", calcNums)
      .then((res) => {
        console.log(res.data);
        setOriginalPayout(res.data.original_payout);
        setBreakEven(res.data.break_even);
        setBreakEvenPayout(res.data.be_payout);
        setEqualReturn(res.data.equal_return);
        setEqualReturnPayout(res.data.er_payout);
      });
  };

  return (
    <div className="layout flex-wrap">
      <div className="setwidth content-center border-r border-l">
        <div className="calc-select flex flex-row">
          <Link to="/calc/payout" className="btn-link calc-btn-select">
            <button className="">Payout</button>
          </Link>
          <Link to="/calc/hedge" className="btn-link">
            <button className="selected">Hedge</button>
          </Link>
          <Link to="/calc/hold" className="btn-link calc-btn-select">
            <button className="">Hold</button>
          </Link>
        </div>
        <div
          className={
            toggleState === 1
              ? "content  active-content CalcTabs h-[1350px]"
              : "content CalcTabs h-[1350px]"
          }
        >
          <h2 className="mt-6 mb-10 title">Hedge Calculator</h2>

          <div className="mb-10">
            <div className="align-center mb-2">Odds Input Type</div>
            <hr className="mb-4" />
            <div className="flex flex-row">
              <div
                className={`${
                  typeState === 1
                    ? "flex-1 btn-link"
                    : "flex-1 btn-link calc-btn-select"
                }`}
              >
                <button
                  className={`${
                    typeState === 1 ? "selected flex-1" : "flex-1"
                  }`}
                  onClick={() => toggleType(1)}
                >
                  American
                </button>
              </div>
              <div
                className={`${
                  typeState === 2
                    ? "flex-1 btn-link"
                    : "flex-1 btn-link calc-btn-select"
                }`}
              >
                <button
                  className={`${
                    typeState === 2 ? "selected flex-1" : "flex-1"
                  }`}
                  onClick={() => toggleType(2)}
                >
                  Decimal
                </button>
              </div>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="form mb-12">
            <label className="">Wager Value</label>
            <input
              type="number"
              placeholder="100"
              required
              name="wagerNum"
              value={wagerNum}
              onChange={(e) => setWager(e.target.value)}
            />
            <label className="">Odds of Wager</label>
            <input
              type="number"
              step="0.01"
              placeholder={typeState === 2 ? "1.5" : "900"}
              required
              name="oddsNum"
              value={oddsNum}
              onChange={(e) => setOdds(e.target.value)}
            />
            <label className="">Odds of Hedge</label>
            <input
              type="number"
              step="0.01"
              placeholder={typeState === 2 ? "1.5" : "900"}
              required
              name="hedgeNum"
              value={hedgeNum}
              onChange={(e) => setHedge(e.target.value)}
            />

            <button className="w-[25%] mt-6 calc-btn font-semibold">
              Calculate
            </button>
          </form>

          <div className="div mb-10 font-normal">
            <h2 className="align-center mb-2">Break Even Hedge Results</h2>
            <hr className="mb-4" />
            <div>
              <h1 className="align-center mb-2 text-[16px]">Total Wagered</h1>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center text-[14px]">
                Original Wager
              </label>
              <label className="flex-1 flex-row align-center text-[14px] highlight">
                Hedge Wager
              </label>
              <label className="flex-1 flex-row align-center text-[14px]">
                Net Wagered
              </label>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center border text-[14px]">
                {wagerNum}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px] highlight">
                {breakEven}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px]">
                {wagerNum
                  ? Math.round(
                      (parseFloat(wagerNum) +
                        parseFloat(breakEven) +
                        Number.EPSILON) *
                        100
                    ) / 100
                  : ""}
              </label>
            </div>
            <div>
              <h1 className="align-center mb-2 text-[16px] mt-4">
                Original Bet Win
              </h1>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center text-[14px]">
                Payout Total
              </label>
              <label className="flex-1 flex-row align-center text-[14px]">
                Net Wagered
              </label>
              <label className="flex-1 flex-row align-center text-[14px]">
                Net Profit
              </label>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center border text-[14px]">
                {originalPayout}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px]">
                {wagerNum
                  ? Math.round(
                      (parseFloat(wagerNum) +
                        parseFloat(breakEven) +
                        Number.EPSILON) *
                        100
                    ) / 100
                  : ""}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px]">
                {wagerNum
                  ? Math.round(
                      originalPayout -
                        ((parseFloat(wagerNum) +
                          parseFloat(breakEven) +
                          Number.EPSILON) *
                          100) /
                          100
                    )
                  : ""}
              </label>
            </div>
            <div>
              <h1 className="align-center mb-2 text-[16px] mt-4">
                Hedge Bet Win
              </h1>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center text-[14px]">
                Payout Total
              </label>
              <label className="flex-1 flex-row align-center text-[14px]">
                Net Wagered
              </label>
              <label className="flex-1 flex-row align-center text-[14px]">
                Net Profit
              </label>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center border text-[14px]">
                {breakEvenPayout}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px]">
                {wagerNum
                  ? Math.round(
                      (parseFloat(wagerNum) +
                        parseFloat(breakEven) +
                        Number.EPSILON) *
                        100
                    ) / 100
                  : ""}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px]">
                0
              </label>
            </div>
          </div>
          <div className="div mb-10 font-normal">
            <h2 className="align-center mb-2">Equal Return Hedge Results</h2>
            <hr className="mb-4" />
            <div>
              <h1 className="align-center mb-2 text-[16px]">Total Wagered</h1>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center text-[14px]">
                Original Wager
              </label>
              <label className="flex-1 flex-row align-center text-[14px] highlight">
                Hedge Wager
              </label>
              <label className="flex-1 flex-row align-center text-[14px]">
                Net Wagered
              </label>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center border text-[14px]">
                {wagerNum}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px] highlight">
                {equalReturn}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px]">
                {wagerNum
                  ? Math.round(
                      (parseFloat(wagerNum) +
                        parseFloat(equalReturn) +
                        Number.EPSILON) *
                        100
                    ) / 100
                  : ""}
              </label>
            </div>
            <div>
              <h1 className="align-center mb-2 text-[16px] mt-4">
                Original Bet Win
              </h1>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center text-[14px]">
                Payout Total
              </label>
              <label className="flex-1 flex-row align-center text-[14px]">
                Net Wagered
              </label>
              <label className="flex-1 flex-row align-center text-[14px]">
                Net Profit
              </label>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center border text-[14px]">
                {originalPayout}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px]">
                {wagerNum
                  ? Math.round(
                      (parseFloat(wagerNum) +
                        parseFloat(equalReturn) +
                        Number.EPSILON) *
                        100
                    ) / 100
                  : ""}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px]">
                {wagerNum
                  ? Math.round(
                      originalPayout -
                        ((parseFloat(wagerNum) +
                          parseFloat(equalReturn) +
                          Number.EPSILON) *
                          100) /
                          100
                    )
                  : ""}
              </label>
            </div>
            <div>
              <h1 className="align-center mb-2 text-[16px] mt-4">
                Hedge Bet Win
              </h1>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center text-[14px]">
                Payout Total
              </label>
              <label className="flex-1 flex-row align-center text-[14px]">
                Net Wagered
              </label>
              <label className="flex-1 flex-row align-center text-[14px]">
                Net Profit
              </label>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center border text-[14px]">
                {equalReturnPayout}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px]">
                {wagerNum
                  ? Math.round(
                      (parseFloat(wagerNum) +
                        parseFloat(equalReturn) +
                        Number.EPSILON) *
                        100
                    ) / 100
                  : ""}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px]">
                {wagerNum
                  ? Math.round(
                      equalReturnPayout -
                        ((parseFloat(wagerNum) +
                          parseFloat(equalReturn) +
                          Number.EPSILON) *
                          100) /
                          100
                    )
                  : ""}
              </label>
            </div>
          </div>
        </div>
      </div>
      <div className="setwidth-info border-r border-l flex flex-col flex-wrap info-sm">
        <div className="calc-select flex flex-row h-[71px] justify-center items-center text-white">
          <h2 className="text-[18px] font-semibold">Information</h2>
        </div>
        <div className="active-content flex">
          <h2 className="text-white p-2 underline underline-offset-4">
            What is a Hedge Bet?
          </h2>
          <p className="text-white p-2 text-[13px]">
            Hedging a bet is a strategy to minimize the losable value or to
            guarantee a profit by placing a second bet that conflicts with the
            first.
          </p>
          <p className="text-white p-2 text-[13px]">
            For example, say you bet placed a wager on Phoenix to win vs the
            Rockets. Phoenix comes out to a very slow start followed by one of
            their star players getting ejected. At this point, even early on,
            it's clear that their odds of winning have dropped considerably. To
            minimize the amount you expect to lose, you could hedge a bet on the
            Rockets. In the given scenario, it's very unlikely you can guarantee
            yourself a profit but you can minimize some of the value lost on the
            original wager.
          </p>
          <h2 className="text-white p-2 underline underline-offset-4">
            What are American Odds?
          </h2>
          <p className="text-white p-2 text-[13px]">
            American Odds are commonly used amongst American bookmakers, as the
            name suggests, and center around a baseline value of $100.
          </p>
          <p className="text-white p-2 text-[13px]">
            When looking at the odds of the favorited team, you might see -145.
            This means that you must risk $145 in order to make $100.
          </p>
          <p className="text-white p-2 text-[13px]">
            For the underdog, you might see +145 (or just 145). This means that
            you will make $145 for a $100 Wager
          </p>
          <h2 className="text-white p-2 underline underline-offset-4">
            What are Decimal Odds?
          </h2>
          <p className="text-white p-2 text-[13px]">
            Decimal Odds are another commonly used system by Bookmakers. With
            decimal odds, odds are presented in a decimal format such as 1.41.
          </p>
          <p className="text-white p-2 text-[13px]">
            Calculating the payout of a wager using decimal odds is fairly
            straightforward. If you wager $100 on Boston at a decimal odds of
            1.41, your return would be $141.
          </p>
          <h2 className="text-white p-2 underline underline-offset-4">
            What are Implied Odds?
          </h2>
          <p className="text-white p-2 text-[13px]">
            Implied Odds are determine based on a Bookmaker's given Decimal or
            American Odds. They represent the likelihood of a team winning, on a
            1-100% scale.
          </p>
          <p className="text-white p-2 text-[13px]">
            For example, Boston has -200 American Odds in their upcoming game vs
            Washington. The Implied Odds of Boston winning the game (at -200)
            are 66.66%.
          </p>
          <p className="text-white p-2 text-[13px]">
            The formula for determining Implied Odds is: (1 / Converted Decimal
            Odds) * 100. (American Odds are Converted to Decimal)
          </p>
          <h2 className="text-white p-2 underline underline-offset-4">
            What is a Break Even Hedge?
          </h2>
          <p className="text-white p-2 text-[13px]">
            As the name implies, the intent of a Break Even Hedge is to
            guarantee that even if your original bet loses, you can still make
            back your original wager by winning your hedged bet.
          </p>
          <h2 className="text-white p-2 underline underline-offset-4">
            What is a Equal Return Hedge?
          </h2>
          <p className="text-white p-2 text-[13px]">
            The intent of an Equal Return Hedge is to guarantee that even
            regardless of whether your original bet or hedged bet outcome, you
            will return the same amount. In doing so, you risk some of your
            potential original payout to reduce overall risk.
          </p>
        </div>
      </div>
    </div>
  );
}

export default Hedge;
