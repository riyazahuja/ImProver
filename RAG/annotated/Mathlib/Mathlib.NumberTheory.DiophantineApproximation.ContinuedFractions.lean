/-- The `n`th convergent of the `GenContFract.of ξ` agrees with `ξ.convergent n`. -/
theorem convs_eq_convergent (ξ : ℝ) (n : ℕ) :
    (GenContFract.of ξ).convs n = ξ.convergent n := by
  /-
    ξ : Real
    n : Nat
    ⊢ Eq ((GenContFract.of ξ).convs n) ↑(ξ.convergent n)
  -/
  induction' n with n ih generalizing ξ
    /-
      case zero
      ξ : Real
      ⊢ Eq ((GenContFract.of ξ).convs 0) ↑(ξ.convergent 0)
    -/
  · simp only [zeroth_conv_eq_h, of_h_eq_floor, convergent_zero, Rat.cast_intCast]
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      ih : ∀ (ξ : Real), Eq ((GenContFract.of ξ).convs n) ↑(ξ.convergent n)
      ξ : Real
      ⊢ Eq ((GenContFract.of ξ).convs (HAdd.hAdd n 1)) ↑(ξ.convergent (HAdd.hAdd n 1))
    -/
  · rw [convs_succ, ih (fract ξ)⁻¹, convergent_succ, one_div]
    /-
      case succ
      n : Nat
      ih : ∀ (ξ : Real), Eq ((GenContFract.of ξ).convs n) ↑(ξ.convergent n)
      ξ : Real
      ⊢ Eq (HAdd.hAdd (↑(Int.floor ξ)) (Inv.inv ↑((Inv.inv (Int.fract ξ)).convergent …
    -/
    norm_cast
    /-
      🎉 no goals
    -/


/-- The main result, *Legendre's Theorem* on rational approximation:
if `ξ` is a real number and `q` is a rational number such that `|ξ - q| < 1/(2*q.den^2)`,
then `q` is a convergent of the continued fraction expansion of `ξ`.
This is the version using `GenContFract.convs`. -/
theorem exists_convs_eq_rat {q : ℚ}
    (h : |ξ - q| < 1 / (2 * (q.den : ℝ) ^ 2)) : ∃ n, (GenContFract.of ξ).convs n = q := by
  /-
    ξ : Real
    q : Rat
    h : LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul 2 (HPow.hPow (↑q.den) …
    ⊢ Exists fun n => Eq ((GenContFract.of ξ).convs n) ↑q
  -/
  obtain ⟨n, hn⟩ := exists_rat_eq_convergent h
  /-
    case intro
    ξ : Real
    q : Rat
    h : LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul 2 (HPow.hPow (↑q.den) …
    n : Nat
    hn : Eq q (ξ.convergent n)
    ⊢ Exists fun n => Eq ((GenContFract.of ξ).convs n) ↑q
  -/
  exact ⟨n, hn.symm ▸ convs_eq_convergent ξ n⟩
  /-
    🎉 no goals
  -/


