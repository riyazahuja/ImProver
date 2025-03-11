/-- The sequence of integrals used for Cartwright's proof of irrationality of `π`. -/
private def I (n : ℕ) (θ : ℝ) : ℝ := ∫ x in (-1)..1, (1 - x ^ 2) ^ n * cos (x * θ)


private lemma I_zero : I 0 θ * θ = 2 * sin θ := by
  /-
    θ : Real
    ⊢ Eq (HMul.hMul (I 0 θ) θ) (HMul.hMul 2 (Real.sin θ))
  -/
  rw [mul_comm, I]
  /-
    θ : Real
    ⊢ Eq (HMul.hMul θ (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub  …
  -/
  simp [mul_integral_comp_mul_right, two_mul]
  /-
    🎉 no goals
  -/


/--
Auxiliary for the proof that `π` is irrational.
While it is most natural to give the recursive formula for `I (n + 2) θ`, as well as give the second
base case of `I 1 θ`, it is in fact more convenient to give the recursive formula for `I (n + 1) θ`
in terms of `I n θ` and `I (n - 1) θ` (note the natural subtraction!).
Despite the usually inconvenient subtraction, this in fact allows deducing both of the above facts
with significantly fewer analysis computations.
In addition, note the `0 ^ n` on the right hand side - this is intentional, and again allows
combining the proof of the "usual" recursion formula and the base case `I 1 θ`.
-/
private lemma recursion' (n : ℕ) :
    I (n + 1) θ * θ ^ 2 = - (2 * 2 * ((n + 1) * (0 ^ n * cos θ))) +
      2 * (n + 1) * (2 * n + 1) * I n θ - 4 * (n + 1) * n * I (n - 1) θ := by
  /-
    θ : Real
    n : Nat
    ⊢ Eq (HMul.hMul (I (HAdd.hAdd n 1) θ) (HPow.hPow θ 2)) (HSub.hSub (HAdd.hAdd ( …
  -/
  rw [I]
  /-
    θ : Real
    n : Nat
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  let f (x : ℝ) : ℝ := 1 - x ^ 2
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  let u₁ (x : ℝ) : ℝ := f x ^ (n + 1)
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  let u₁' (x : ℝ) : ℝ := - (2 * (n + 1) * x * f x ^ n)
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  let v₁ (x : ℝ) : ℝ := sin (x * θ)
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  let v₁' (x : ℝ) : ℝ := cos (x * θ) * θ
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  let u₂ (x : ℝ) : ℝ := x * (f x) ^ n
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  let u₂' (x : ℝ) : ℝ := (f x) ^ n - 2 * n * x ^ 2 * (f x) ^ (n - 1)
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  let v₂ (x : ℝ) : ℝ := cos (x * θ)
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  let v₂' (x : ℝ) : ℝ := -sin (x * θ) * θ
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  have hfd : Continuous f := by fun_prop
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  have hu₁d : Continuous u₁' := by fun_prop
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  have hv₁d : Continuous v₁' := by fun_prop
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  have hu₂d : Continuous u₂' := by fun_prop
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    hu₂d : Continuous u₂'
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  have hv₂d : Continuous v₂' := by fun_prop
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    hu₂d : Continuous u₂'
    hv₂d : Continuous v₂'
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  have hu₁_eval_one : u₁ 1 = 0 := by simp only [u₁, f]; simp
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    hu₂d : Continuous u₂'
    hv₂d : Continuous v₂'
    hu₁_eval_one : Eq (u₁ 1) 0
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  have hu₁_eval_neg_one : u₁ (-1) = 0 := by simp only [u₁, f]; simp
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    hu₂d : Continuous u₂'
    hv₂d : Continuous v₂'
    hu₁_eval_one : Eq (u₁ 1) 0
    hu₁_eval_neg_one : Eq (u₁ (-1)) 0
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  have t : u₂ 1 * v₂ 1 - u₂ (-1) * v₂ (-1) = 2 * (0 ^ n * cos θ) := by simp [u₂, v₂, f, ← two_mul]
  have hf (x) : HasDerivAt f (- 2 * x) x := by
    convert (hasDerivAt_pow 2 x).const_sub 1 using 1
    simp
  have hu₁ (x) : HasDerivAt u₁ (u₁' x) x := by
    convert (hf x).pow _ using 1
    simp only [Nat.add_succ_sub_one, u₁', Nat.cast_add_one]
    ring
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    hu₂d : Continuous u₂'
    hv₂d : Continuous v₂'
    hu₁_eval_one : Eq (u₁ 1) 0
    hu₁_eval_neg_one : Eq (u₁ (-1)) 0
    t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
    hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
    hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  have hv₁ (x) : HasDerivAt v₁ (v₁' x) x := (hasDerivAt_mul_const θ).sin
  have hu₂ (x) : HasDerivAt u₂ (u₂' x) x := by
    convert (hasDerivAt_id' x).mul ((hf x).pow _) using 1
    simp only [u₂']
    ring
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    hu₂d : Continuous u₂'
    hv₂d : Continuous v₂'
    hu₁_eval_one : Eq (u₁ 1) 0
    hu₁_eval_neg_one : Eq (u₁ (-1)) 0
    t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
    hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
    hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
    hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
    hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  have hv₂ (x) : HasDerivAt v₂ (v₂' x) x := (hasDerivAt_mul_const θ).cos
  /-
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    hu₂d : Continuous u₂'
    hv₂d : Continuous v₂'
    hu₁_eval_one : Eq (u₁ 1) 0
    hu₁_eval_neg_one : Eq (u₁ (-1)) 0
    t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
    hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
    hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
    hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
    hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
    hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
  -/
  convert_to (∫ (x : ℝ) in (-1)..1, u₁ x * v₁' x) * θ = _ using 1
    /-
      case h.e'_2
      θ : Real
      n : Nat
      f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
      u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
      u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
      v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
      v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
      u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
      u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
      v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
      v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
      hfd : Continuous f
      hu₁d : Continuous u₁'
      hv₁d : Continuous v₁'
      hu₂d : Continuous u₂'
      hv₂d : Continuous v₂'
      hu₁_eval_one : Eq (u₁ 1) 0
      hu₁_eval_neg_one : Eq (u₁ (-1)) 0
      t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
      hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
      hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
      hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
      hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
      hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
      ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HSub.hSub 1  …
    -/
  · simp_rw [u₁, v₁', f, ← intervalIntegral.integral_mul_const, sq θ, mul_assoc]
    /-
      🎉 no goals
    -/
  rw [integral_mul_deriv_eq_deriv_mul (fun x _ => hu₁ x) (fun x _ => hv₁ x)
    (hu₁d.intervalIntegrable _ _) (hv₁d.intervalIntegrable _ _), hu₁_eval_one, hu₁_eval_neg_one,
    zero_mul, zero_mul, sub_zero, zero_sub, ← integral_neg, ← integral_mul_const]
  /-
    case convert_2
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    hu₂d : Continuous u₂'
    hv₂d : Continuous v₂'
    hu₁_eval_one : Eq (u₁ 1) 0
    hu₁_eval_neg_one : Eq (u₁ (-1)) 0
    t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
    hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
    hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
    hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
    hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
    hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (Neg.neg (HMul.hMul (u₁' x) (v₁ x)) …
  -/
  convert_to ((-2 : ℝ) * (n + 1)) * ∫ (x : ℝ) in (-1)..1, (u₂ x * v₂' x) = _ using 1
    /-
      case h.e'_2
      θ : Real
      n : Nat
      f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
      u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
      u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
      v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
      v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
      u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
      u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
      v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
      v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
      hfd : Continuous f
      hu₁d : Continuous u₁'
      hv₁d : Continuous v₁'
      hu₂d : Continuous u₂'
      hv₂d : Continuous v₂'
      hu₁_eval_one : Eq (u₁ 1) 0
      hu₁_eval_neg_one : Eq (u₁ (-1)) 0
      t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
      hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
      hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
      hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
      hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
      hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
      ⊢ Eq (intervalIntegral (fun x => HMul.hMul (Neg.neg (HMul.hMul (u₁' x) (v₁ x)) …
    -/
  · rw [← integral_const_mul]
    /-
      case h.e'_2
      θ : Real
      n : Nat
      f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
      u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
      u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
      v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
      v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
      u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
      u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
      v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
      v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
      hfd : Continuous f
      hu₁d : Continuous u₁'
      hv₁d : Continuous v₁'
      hu₂d : Continuous u₂'
      hv₂d : Continuous v₂'
      hu₁_eval_one : Eq (u₁ 1) 0
      hu₁_eval_neg_one : Eq (u₁ (-1)) 0
      t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
      hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
      hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
      hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
      hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
      hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
      ⊢ Eq (intervalIntegral (fun x => HMul.hMul (Neg.neg (HMul.hMul (u₁' x) (v₁ x)) …
    -/
    congr 1 with x
    /-
      case h.e'_2.e_f.h
      θ : Real
      n : Nat
      f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
      u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
      u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
      v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
      v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
      u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
      u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
      v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
      v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
      hfd : Continuous f
      hu₁d : Continuous u₁'
      hv₁d : Continuous v₁'
      hu₂d : Continuous u₂'
      hv₂d : Continuous v₂'
      hu₁_eval_one : Eq (u₁ 1) 0
      hu₁_eval_neg_one : Eq (u₁ (-1)) 0
      t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
      hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
      hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
      hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
      hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
      hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
      x : Real
      ⊢ Eq (HMul.hMul (Neg.neg (HMul.hMul (u₁' x) (v₁ x))) θ) (HMul.hMul (HMul.hMul  …
    -/
    dsimp [u₁', v₁, u₂, v₂']
    /-
      case h.e'_2.e_f.h
      θ : Real
      n : Nat
      f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
      u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
      u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
      v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
      v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
      u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
      u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
      v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
      v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
      hfd : Continuous f
      hu₁d : Continuous u₁'
      hv₁d : Continuous v₁'
      hu₂d : Continuous u₂'
      hv₂d : Continuous v₂'
      hu₁_eval_one : Eq (u₁ 1) 0
      hu₁_eval_neg_one : Eq (u₁ (-1)) 0
      t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
      hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
      hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
      hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
      hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
      hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
      x : Real
      ⊢ Eq (HMul.hMul (Neg.neg (HMul.hMul (Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul  …
    -/
    ring
    /-
      🎉 no goals
    -/
  rw [integral_mul_deriv_eq_deriv_mul (fun x _ => hu₂ x) (fun x _ => hv₂ x)
    (hu₂d.intervalIntegrable _ _) (hv₂d.intervalIntegrable _ _),
    mul_sub, t, neg_mul, neg_mul, neg_mul, sub_neg_eq_add]
  have (x) : u₂' x = (2 * n + 1) * f x ^ n - 2 * n * f x ^ (n - 1) := by
    cases n with
    | zero => simp [u₂']
    | succ n => ring!
  /-
    case convert_2.convert_2
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    hu₂d : Continuous u₂'
    hv₂d : Continuous v₂'
    hu₁_eval_one : Eq (u₁ 1) 0
    hu₁_eval_neg_one : Eq (u₁ (-1)) 0
    t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
    hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
    hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
    hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
    hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
    hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
    this : ∀ (x : Real), Eq (u₂' x) (HSub.hSub (HMul.hMul (HAdd.hAdd (HMul.hMul 2  …
    ⊢ Eq (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) (HMul.hMu …
  -/
  simp_rw [this, sub_mul, mul_assoc _ _ (v₂ _)]
  /-
    case convert_2.convert_2
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    hu₂d : Continuous u₂'
    hv₂d : Continuous v₂'
    hu₁_eval_one : Eq (u₁ 1) 0
    hu₁_eval_neg_one : Eq (u₁ (-1)) 0
    t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
    hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
    hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
    hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
    hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
    hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
    this : ∀ (x : Real), Eq (u₂' x) (HSub.hSub (HMul.hMul (HAdd.hAdd (HMul.hMul 2  …
    ⊢ Eq (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) (HMul.hMu …
  -/
  have : Continuous v₂ := by fun_prop
  /-
    case convert_2.convert_2
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    hu₂d : Continuous u₂'
    hv₂d : Continuous v₂'
    hu₁_eval_one : Eq (u₁ 1) 0
    hu₁_eval_neg_one : Eq (u₁ (-1)) 0
    t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
    hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
    hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
    hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
    hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
    hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
    this✝ : ∀ (x : Real), Eq (u₂' x) (HSub.hSub (HMul.hMul (HAdd.hAdd (HMul.hMul 2 …
    this : Continuous v₂
    ⊢ Eq (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) (HMul.hMu …
  -/
  rw [mul_mul_mul_comm, integral_sub, mul_sub, add_sub_assoc]
    /-
      case convert_2.convert_2
      θ : Real
      n : Nat
      f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
      u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
      u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
      v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
      v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
      u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
      u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
      v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
      v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
      hfd : Continuous f
      hu₁d : Continuous u₁'
      hv₁d : Continuous v₁'
      hu₂d : Continuous u₂'
      hv₂d : Continuous v₂'
      hu₁_eval_one : Eq (u₁ 1) 0
      hu₁_eval_neg_one : Eq (u₁ (-1)) 0
      t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
      hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
      hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
      hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
      hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
      hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
      this✝ : ∀ (x : Real), Eq (u₂' x) (HSub.hSub (HMul.hMul (HAdd.hAdd (HMul.hMul 2 …
      this : Continuous v₂
      ⊢ Eq (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul 2 2) (HMul.hMul (HAdd.hAdd (↑n) …
    -/
  · congr 1
    /-
      case convert_2.convert_2.e_a
      θ : Real
      n : Nat
      f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
      u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
      u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
      v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
      v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
      u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
      u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
      v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
      v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
      hfd : Continuous f
      hu₁d : Continuous u₁'
      hv₁d : Continuous v₁'
      hu₂d : Continuous u₂'
      hv₂d : Continuous v₂'
      hu₁_eval_one : Eq (u₁ 1) 0
      hu₁_eval_neg_one : Eq (u₁ (-1)) 0
      t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
      hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
      hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
      hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
      hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
      hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
      this✝ : ∀ (x : Real), Eq (u₂' x) (HSub.hSub (HMul.hMul (HAdd.hAdd (HMul.hMul 2 …
      this : Continuous v₂
      ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) (intervalIntegral  …
    -/
    simp_rw [integral_const_mul]
    /-
      case convert_2.convert_2.e_a
      θ : Real
      n : Nat
      f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
      u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
      u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
      v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
      v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
      u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
      u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
      v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
      v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
      hfd : Continuous f
      hu₁d : Continuous u₁'
      hv₁d : Continuous v₁'
      hu₂d : Continuous u₂'
      hv₂d : Continuous v₂'
      hu₁_eval_one : Eq (u₁ 1) 0
      hu₁_eval_neg_one : Eq (u₁ (-1)) 0
      t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
      hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
      hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
      hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
      hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
      hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
      this✝ : ∀ (x : Real), Eq (u₂' x) (HSub.hSub (HMul.hMul (HAdd.hAdd (HMul.hMul 2 …
      this : Continuous v₂
      ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul 2 (HAdd.hAdd (↑n) 1)) (HMul.hMul (HAdd.h …
    -/
    ring!
    /-
      🎉 no goals
    -/
  /-
    case convert_2.convert_2.hf
    θ : Real
    n : Nat
    f : Real → Real := fun x => HSub.hSub 1 (HPow.hPow x 2)
    u₁ : Real → Real := fun x => HPow.hPow (f x) (HAdd.hAdd n 1)
    u₁' : Real → Real := fun x => Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd …
    v₁ : Real → Real := fun x => Real.sin (HMul.hMul x θ)
    v₁' : Real → Real := fun x => HMul.hMul (Real.cos (HMul.hMul x θ)) θ
    u₂ : Real → Real := fun x => HMul.hMul x (HPow.hPow (f x) n)
    u₂' : Real → Real := fun x => HSub.hSub (HPow.hPow (f x) n) (HMul.hMul (HMul.h …
    v₂ : Real → Real := fun x => Real.cos (HMul.hMul x θ)
    v₂' : Real → Real := fun x => HMul.hMul (Neg.neg (Real.sin (HMul.hMul x θ))) θ
    hfd : Continuous f
    hu₁d : Continuous u₁'
    hv₁d : Continuous v₁'
    hu₂d : Continuous u₂'
    hv₂d : Continuous v₂'
    hu₁_eval_one : Eq (u₁ 1) 0
    hu₁_eval_neg_one : Eq (u₁ (-1)) 0
    t : Eq (HSub.hSub (HMul.hMul (u₂ 1) (v₂ 1)) (HMul.hMul (u₂ (-1)) (v₂ (-1)))) ( …
    hf : ∀ (x : Real), HasDerivAt f (HMul.hMul (-2) x) x
    hu₁ : ∀ (x : Real), HasDerivAt u₁ (u₁' x) x
    hv₁ : ∀ (x : Real), HasDerivAt v₁ (v₁' x) x
    hu₂ : ∀ (x : Real), HasDerivAt u₂ (u₂' x) x
    hv₂ : ∀ (x : Real), HasDerivAt v₂ (v₂' x) x
    this✝ : ∀ (x : Real), Eq (u₂' x) (HSub.hSub (HMul.hMul (HAdd.hAdd (HMul.hMul 2 …
    this : Continuous v₂
    ⊢ IntervalIntegrable (fun x => HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑n) 1) (HMul. …
  -/
  all_goals exact Continuous.intervalIntegrable (by fun_prop) _ _
  /-
    🎉 no goals
  -/


/--
Auxiliary for the proof that `π` is irrational.
The recursive formula for `I (n + 2) θ * θ ^ 2` in terms of `I n θ` and `I (n + 1) θ`.
-/
private lemma recursion (n : ℕ) :
    I (n + 2) θ * θ ^ 2 =
      2 * (n + 2) * (2 * n + 3) * I (n + 1) θ - 4 * (n + 2) * (n + 1) * I n θ := by
  /-
    θ : Real
    n : Nat
    ⊢ Eq (HMul.hMul (I (HAdd.hAdd n 2) θ) (HPow.hPow θ 2)) (HSub.hSub (HMul.hMul ( …
  -/
  rw [recursion' (n + 1)]
  /-
    θ : Real
    n : Nat
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Neg.neg (HMul.hMul (HMul.hMul 2 2) (HMul.hMul (HAd …
  -/
  simp
  /-
    θ : Real
    n : Nat
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul (HMul.hMul 2 (HAdd.hAdd (HAdd.hAdd (↑n)  …
  -/
  ring!
  /-
    🎉 no goals
  -/


/--
Auxiliary for the proof that `π` is irrational.
The second base case for the induction on `n`, giving an explicit formula for `I 1 θ`.
-/
private lemma I_one : I 1 θ * θ ^ 3 = 4 * sin θ - 4 * θ * cos θ := by
  /-
    θ : Real
    ⊢ Eq (HMul.hMul (I 1 θ) (HPow.hPow θ 3)) (HSub.hSub (HMul.hMul 4 (Real.sin θ)) …
  -/
  rw [_root_.pow_succ, ← mul_assoc, recursion' 0, sub_mul, add_mul, mul_assoc _ (I 0 θ), I_zero]
  /-
    θ : Real
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (Neg.neg (HMul.hMul (HMul.hMul 2 2) (HMu …
  -/
  ring
  /-
    🎉 no goals
  -/


/--
Auxiliary for the proof that `π` is irrational.
The first of the two integer-coefficient polynomials that describe the behaviour of the
sequence of integrals `I`.
While not given in the informal proof, these are easy to deduce from the recursion formulae.
-/
private def sinPoly : ℕ → ℤ[X]
  | 0 => C 2
  | 1 => C 4
  | (n+2) => ((2 : ℤ) * (2 * n + 3)) • sinPoly (n + 1) + monomial 2 (-4) * sinPoly n


/--
Auxiliary for the proof that `π` is irrational.
The second of the two integer-coefficient polynomials that describe the behaviour of the
sequence of integrals `I`.
While not given in the informal proof, these are easy to deduce from the recursion formulae.
-/
private def cosPoly : ℕ → ℤ[X]
  | 0 => 0
  | 1 => monomial 1 (-4)
  | (n+2) => ((2 : ℤ) * (2 * n + 3)) • cosPoly (n + 1) + monomial 2 (-4) * cosPoly n


/--
Auxiliary for the proof that `π` is irrational.
Prove a degree bound for `sinPoly n` by induction. Note this is where we find the value in an
explicit description of `sinPoly`.
-/
private lemma sinPoly_natDegree_le : ∀ n : ℕ, (sinPoly n).natDegree ≤ n
            /-
              ⊢ LE.le (sinPoly 0).natDegree 0
            -/
  | 0 => by simp [sinPoly]
            /-
              🎉 no goals
            -/
            /-
              ⊢ LE.le (sinPoly 1).natDegree 1
            -/
  | 1 => by simp only [natDegree_C, mul_one, zero_le', sinPoly]
            /-
              🎉 no goals
            -/
  | n + 2 => by
      /-
        n : Nat
        ⊢ LE.le (sinPoly (HAdd.hAdd n 2)).natDegree (HAdd.hAdd n 2)
      -/
      rw [sinPoly]
      /-
        n : Nat
        ⊢ LE.le (HAdd.hAdd (HSMul.hSMul (HMul.hMul 2 (HAdd.hAdd (HMul.hMul 2 ↑n) 3)) ( …
      -/
      refine natDegree_add_le_of_degree_le ((natDegree_smul_le _ _).trans ?_) ?_
        /-
          case refine_1
          n : Nat
          ⊢ LE.le (sinPoly (HAdd.hAdd n 1)).natDegree (HAdd.hAdd n 2)
        -/
      · exact (sinPoly_natDegree_le (n + 1)).trans (by simp)
        /-
          🎉 no goals
        -/
      /-
        case refine_2
        n : Nat
        ⊢ LE.le (HMul.hMul ((Polynomial.monomial 2) (-4)) (sinPoly n)).natDegree (HAdd …
      -/
      refine natDegree_mul_le.trans ?_
      /-
        case refine_2
        n : Nat
        ⊢ LE.le (HAdd.hAdd ((Polynomial.monomial 2) (-4)).natDegree (sinPoly n).natDeg …
      -/
      simpa [add_comm 2] using sinPoly_natDegree_le n
      /-
        🎉 no goals
      -/


/--
Auxiliary for the proof that `π` is irrational.
Prove a degree bound for `cosPoly n` by induction. Note this is where we find the value in an
explicit description of `cosPoly`.
-/
private lemma cosPoly_natDegree_le : ∀ n : ℕ, (cosPoly n).natDegree ≤ n
            /-
              ⊢ LE.le (cosPoly 0).natDegree 0
            -/
  | 0 => by simp [cosPoly]
            /-
              🎉 no goals
            -/
                                             /-
                                               ⊢ LE.le 1 1
                                             -/
  | 1 => (natDegree_monomial_le _).trans (by simp)
                                             /-
                                               🎉 no goals
                                             -/
  | n + 2 => by
      /-
        n : Nat
        ⊢ LE.le (cosPoly (HAdd.hAdd n 2)).natDegree (HAdd.hAdd n 2)
      -/
      rw [cosPoly]
      /-
        n : Nat
        ⊢ LE.le (HAdd.hAdd (HSMul.hSMul (HMul.hMul 2 (HAdd.hAdd (HMul.hMul 2 ↑n) 3)) ( …
      -/
      refine natDegree_add_le_of_degree_le ((natDegree_smul_le _ _).trans ?_) ?_
        /-
          case refine_1
          n : Nat
          ⊢ LE.le (cosPoly (HAdd.hAdd n 1)).natDegree (HAdd.hAdd n 2)
        -/
      · exact (cosPoly_natDegree_le (n + 1)).trans (by simp)
        /-
          🎉 no goals
        -/
      /-
        case refine_2
        n : Nat
        ⊢ LE.le (HMul.hMul ((Polynomial.monomial 2) (-4)) (cosPoly n)).natDegree (HAdd …
      -/
      exact natDegree_mul_le.trans (by simp [add_comm 2, cosPoly_natDegree_le n])
      /-
        🎉 no goals
      -/


/--
Auxiliary for the proof that `π` is irrational.
The key lemma: the sequence of integrals `I` can be written as a linear combination of `sin` and
`cos`, with coefficients given by the polynomials `sinPoly` and `cosPoly`.
-/
private lemma sinPoly_add_cosPoly_eval (θ : ℝ) :
    ∀ n : ℕ,
      I n θ * θ ^ (2 * n + 1) = n ! * ((sinPoly n).eval₂ (Int.castRingHom _) θ * sin θ +
        (cosPoly n).eval₂ (Int.castRingHom _) θ * cos θ)
            /-
              θ : Real
              ⊢ Eq (HMul.hMul (I 0 θ) (HPow.hPow θ (HAdd.hAdd (HMul.hMul 2 0) 1))) (HMul.hMu …
            -/
  | 0 => by simp [sinPoly, cosPoly, I_zero]
            /-
              🎉 no goals
            -/
            /-
              θ : Real
              ⊢ Eq (HMul.hMul (I 1 θ) (HPow.hPow θ (HAdd.hAdd (HMul.hMul 2 1) 1))) (HMul.hMu …
            -/
  | 1 => by simp [I_one, sinPoly, cosPoly, sub_eq_add_neg]
            /-
              🎉 no goals
            -/
  | n + 2 => by
      calc I (n + 2) θ * θ ^ (2 * (n + 2) + 1) = I (n + 2) θ * θ ^ 2 * θ ^ (2 * n + 3) := by ring
        _ = 2 * (n + 2) * (2 * n + 3) * (I (n + 1) θ * θ ^ (2 * (n + 1) + 1)) -
            4 * (n + 2) * (n + 1) * θ ^ 2 * (I n θ * θ ^ (2 * n + 1)) := by rw [recursion]; ring
        _ = _ := by simp [sinPoly_add_cosPoly_eval, sinPoly, cosPoly, Nat.factorial_succ]; ring


/--
Auxiliary for the proof that `π` is irrational.
For a polynomial `p` with natural degree `≤ k` and integer coefficients, evaluating `p` at a
rational `a / b` gives a rational of the form `z / b ^ k`.
TODO: should this be moved elsewhere? It uses none of the pi-specific definitions.
-/
private lemma is_integer {p : ℤ[X]} (a b : ℤ) {k : ℕ} (hp : p.natDegree ≤ k) :
    ∃ z : ℤ, p.eval₂ (Int.castRingHom ℝ) (a / b) * b ^ k = z := by
  /-
    p : Polynomial Int
    a b : Int
    k : Nat
    hp : LE.le p.natDegree k
    ⊢ Exists fun z => Eq (HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real) (HDiv …
  -/
  rcases eq_or_ne b 0 with rfl | hb
    /-
      case inl
      p : Polynomial Int
      a : Int
      k : Nat
      hp : LE.le p.natDegree k
      ⊢ Exists fun z => Eq (HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real) (HDiv …
    -/
  · rcases k.eq_zero_or_pos with rfl | hk
      /-
        case inl.inl
        p : Polynomial Int
        a : Int
        hp : LE.le p.natDegree 0
        ⊢ Exists fun z => Eq (HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real) (HDiv …
      -/
    · exact ⟨p.coeff 0, by simp⟩
      /-
        🎉 no goals
      -/
    /-
      case inl.inr
      p : Polynomial Int
      a : Int
      k : Nat
      hp : LE.le p.natDegree k
      hk : GT.gt k 0
      ⊢ Exists fun z => Eq (HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real) (HDiv …
    -/
    exact ⟨0, by simp [hk.ne']⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Polynomial Int
    a b : Int
    k : Nat
    hp : LE.le p.natDegree k
    hb : Ne b 0
    ⊢ Exists fun z => Eq (HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real) (HDiv …
  -/
  refine ⟨∑ i in p.support, p.coeff i * a ^ i * b ^ (k - i), ?_⟩
  /-
    case inr
    p : Polynomial Int
    a b : Int
    k : Nat
    hp : LE.le p.natDegree k
    hb : Ne b 0
    ⊢ Eq (HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real) (HDiv.hDiv ↑a ↑b) p)  …
  -/
  conv => lhs; rw [← sum_monomial_eq p]
  /-
    case inr
    p : Polynomial Int
    a b : Int
    k : Nat
    hp : LE.le p.natDegree k
    hb : Ne b 0
    ⊢ Eq (HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real) (HDiv.hDiv ↑a ↑b) (p. …
  -/
  rw [eval₂_sum, sum, Finset.sum_mul, Int.cast_sum]
  /-
    case inr
    p : Polynomial Int
    a b : Int
    k : Nat
    hp : LE.le p.natDegree k
    hb : Ne b 0
    ⊢ Eq (p.support.sum fun i => HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real …
  -/
  simp only [eval₂_monomial, eq_intCast, div_pow, Int.cast_mul, Int.cast_pow]
  /-
    case inr
    p : Polynomial Int
    a b : Int
    k : Nat
    hp : LE.le p.natDegree k
    hb : Ne b 0
    ⊢ Eq (p.support.sum fun x => HMul.hMul (HMul.hMul (↑(p.coeff x)) (HDiv.hDiv (H …
  -/
  refine Finset.sum_congr rfl (fun i hi => ?_)
  /-
    case inr
    p : Polynomial Int
    a b : Int
    k : Nat
    hp : LE.le p.natDegree k
    hb : Ne b 0
    i : Nat
    hi : Membership.mem p.support i
    ⊢ Eq (HMul.hMul (HMul.hMul (↑(p.coeff i)) (HDiv.hDiv (HPow.hPow (↑a) i) (HPow. …
  -/
  have ik := (le_natDegree_of_mem_supp i hi).trans hp
  rw [mul_assoc, div_mul_comm, ← Int.cast_pow, ← Int.cast_pow, ← Int.cast_pow,
    ← pow_sub_mul_pow b ik, ← Int.cast_div_charZero, Int.mul_ediv_cancel _ (pow_ne_zero _ hb),
    ← mul_assoc, mul_right_comm, ← Int.cast_pow]
  /-
    case inr
    p : Polynomial Int
    a b : Int
    k : Nat
    hp : LE.le p.natDegree k
    hb : Ne b 0
    i : Nat
    hi : Membership.mem p.support i
    ik : LE.le i k
    ⊢ Dvd.dvd (HPow.hPow b i) (HMul.hMul (HPow.hPow b (HSub.hSub k i)) (HPow.hPow  …
  -/
  exact dvd_mul_left _ _
  /-
    🎉 no goals
  -/


/--
Auxiliary for the proof that `π` is irrational.
The integrand in the definition of `I` is nonnegative and takes a positive value at least one point,
so the integral is positive.
-/
private lemma I_pos : 0 < I n (π / 2) := by
  /-
    n : Nat
    ⊢ LT.lt 0 (I n (HDiv.hDiv Real.pi 2))
  -/
  refine integral_pos (by norm_num) (Continuous.continuousOn (by continuity)) ?_ ⟨0, by simp⟩
  /-
    n : Nat
    ⊢ ∀ (x : Real), Membership.mem (Set.Ioc (-1) 1) x → LE.le 0 (HMul.hMul (HPow.h …
  -/
  refine fun x hx => mul_nonneg (pow_nonneg ?_ _) ?_
    /-
      case refine_1
      n : Nat
      x : Real
      hx : Membership.mem (Set.Ioc (-1) 1) x
      ⊢ LE.le 0 (HSub.hSub 1 (HPow.hPow x 2))
    -/
  · rw [sub_nonneg, sq_le_one_iff_abs_le_one, abs_le]
    /-
      case refine_1
      n : Nat
      x : Real
      hx : Membership.mem (Set.Ioc (-1) 1) x
      ⊢ And (LE.le (-1) x) (LE.le x 1)
    -/
    exact ⟨hx.1.le, hx.2⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    n : Nat
    x : Real
    hx : Membership.mem (Set.Ioc (-1) 1) x
    ⊢ LE.le 0 (Real.cos (HMul.hMul x (HDiv.hDiv Real.pi 2)))
  -/
  refine cos_nonneg_of_neg_pi_div_two_le_of_le ?_ ?_ <;>
  /-
    case refine_2.refine_1
    n : Nat
    x : Real
    hx : Membership.mem (Set.Ioc (-1) 1) x
    ⊢ LE.le (Neg.neg (HDiv.hDiv Real.pi 2)) (HMul.hMul x (HDiv.hDiv Real.pi 2))
  -/
  /-
    🎉 no goals
  -/
  nlinarith [hx.1, hx.2, pi_pos]
  /-
    🎉 no goals
  -/


/--
Auxiliary for the proof that `π` is irrational.
The integrand in the definition of `I` is bounded by 1 and the interval has length 2, so the
integral is bounded above by `2`.
-/
private lemma I_le (n : ℕ) : I n (π / 2) ≤ 2 := by
  /-
    n : Nat
    ⊢ LE.le (I n (HDiv.hDiv Real.pi 2)) 2
  -/
  rw [← norm_of_nonneg I_pos.le]
  /-
    n : Nat
    ⊢ LE.le (Norm.norm (I n (HDiv.hDiv Real.pi 2))) 2
  -/
  refine (norm_integral_le_of_norm_le_const ?_).trans (show (1 : ℝ) * _ ≤ _ by norm_num)
  /-
    n : Nat
    ⊢ ∀ (x : Real), Membership.mem (Set.uIoc (-1) 1) x → LE.le (Norm.norm (HMul.hM …
  -/
  intros x hx
  /-
    n : Nat
    x : Real
    hx : Membership.mem (Set.uIoc (-1) 1) x
    ⊢ LE.le (Norm.norm (HMul.hMul (HPow.hPow (HSub.hSub 1 (HPow.hPow x 2)) n) (Rea …
  -/
  simp only [uIoc_of_le, neg_le_self_iff, zero_le_one, mem_Ioc] at hx
  /-
    n : Nat
    x : Real
    hx : And (LT.lt (-1) x) (LE.le x 1)
    ⊢ LE.le (Norm.norm (HMul.hMul (HPow.hPow (HSub.hSub 1 (HPow.hPow x 2)) n) (Rea …
  -/
  rw [norm_eq_abs, abs_mul, abs_pow]
  /-
    n : Nat
    x : Real
    hx : And (LT.lt (-1) x) (LE.le x 1)
    ⊢ LE.le (HMul.hMul (HPow.hPow (abs (HSub.hSub 1 (HPow.hPow x 2))) n) (abs (Rea …
  -/
  refine mul_le_one₀ (pow_le_one₀ (abs_nonneg _) ?_) (abs_nonneg _) (abs_cos_le_one _)
  /-
    n : Nat
    x : Real
    hx : And (LT.lt (-1) x) (LE.le x 1)
    ⊢ LE.le (abs (HSub.hSub 1 (HPow.hPow x 2))) 1
  -/
  rw [abs_le]
  /-
    n : Nat
    x : Real
    hx : And (LT.lt (-1) x) (LE.le x 1)
    ⊢ And (LE.le (-1) (HSub.hSub 1 (HPow.hPow x 2))) (LE.le (HSub.hSub 1 (HPow.hPo …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> nlinarith
                  /-
                    🎉 no goals
                  -/


/--
Auxiliary for the proof that `π` is irrational.
For any real `a`, we have that `a ^ (2n+1) / n!` tends to `0` as `n → ∞`.  This is just a
reformulation of tendsto_pow_div_factorial_atTop, which asserts the same for `a ^ n / n!`
-/
private lemma tendsto_pow_div_factorial_at_top_aux (a : ℝ) :
    Tendsto (fun n => (a : ℝ) ^ (2 * n + 1) / n !) atTop (nhds 0) := by
  /-
    a : Real
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (HPow.hPow a (HAdd.hAdd (HMul.hMul 2 n) 1 …
  -/
  rw [← mul_zero a]
  /-
    a : Real
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (HPow.hPow a (HAdd.hAdd (HMul.hMul 2 n) 1 …
  -/
  refine ((FloorSemiring.tendsto_pow_div_factorial_atTop (a ^ 2)).const_mul a).congr (fun x => ?_)
  /-
    a : Real
    x : Nat
    ⊢ Eq (HMul.hMul a (HDiv.hDiv (HPow.hPow (HPow.hPow a 2) x) ↑x.factorial)) (HDi …
  -/
  rw [← pow_mul, mul_div_assoc', _root_.pow_succ']
  /-
    🎉 no goals
  -/


/-- If `x` is rational, it can be written as `a / b` with `a : ℤ` and `b : ℕ` satisfying `b > 0`. -/
private lemma not_irrational_exists_rep {x : ℝ} :
    ¬Irrational x → ∃ (a : ℤ) (b : ℕ), 0 < b ∧ x = a / b := by
  /-
    x : Real
    ⊢ Not (Irrational x) → Exists fun a => Exists fun b => And (LT.lt 0 b) (Eq x ( …
  -/
  rw [Irrational, not_not, mem_range]
  /-
    x : Real
    ⊢ (Exists fun y => Eq (↑y) x) → Exists fun a => Exists fun b => And (LT.lt 0 b …
  -/
  rintro ⟨q, rfl⟩
  /-
    case intro
    q : Rat
    ⊢ Exists fun a => Exists fun b => And (LT.lt 0 b) (Eq (↑q) (HDiv.hDiv ↑a ↑b))
  -/
  exact ⟨q.num, q.den, q.pos, by exact_mod_cast (Rat.num_div_den _).symm⟩
  /-
    🎉 no goals
  -/


@[simp] theorem irrational_pi : Irrational π := by
  /-
    ⊢ Irrational Real.pi
  -/
  apply Irrational.of_div_nat 2
  /-
    ⊢ Irrational (HDiv.hDiv Real.pi ↑2)
  -/
  rw [Nat.cast_two]
  /-
    ⊢ Irrational (HDiv.hDiv Real.pi 2)
  -/
  by_contra h'
  /-
    h' : Not (Irrational (HDiv.hDiv Real.pi 2))
    ⊢ False
  -/
  obtain ⟨a, b, hb, h⟩ := not_irrational_exists_rep h'
  have ha : (0 : ℝ) < a := by
    have : 0 < (a : ℝ) / b := h ▸ pi_div_two_pos
    rwa [lt_div_iff₀ (by positivity), zero_mul] at this
  /-
    case intro.intro.intro
    h' : Not (Irrational (HDiv.hDiv Real.pi 2))
    a : Int
    b : Nat
    hb : LT.lt 0 b
    h : Eq (HDiv.hDiv Real.pi 2) (HDiv.hDiv ↑a ↑b)
    ha : LT.lt 0 ↑a
    ⊢ False
  -/
  have k (n : ℕ) : 0 < (a : ℝ) ^ (2 * n + 1) / n ! := by positivity
  have j : ∀ᶠ n : ℕ in atTop, (a : ℝ) ^ (2 * n + 1) / n ! * I n (π / 2) < 1 := by
    have := (tendsto_pow_div_factorial_at_top_aux a).eventually_lt_const
      (show (0 : ℝ) < 1 / 2 by norm_num)
    filter_upwards [this] with n hn
    rw [lt_div_iff₀ (zero_lt_two : (0 : ℝ) < 2)] at hn
    exact hn.trans_le' (mul_le_mul_of_nonneg_left (I_le _) (by positivity))
  /-
    case intro.intro.intro
    h' : Not (Irrational (HDiv.hDiv Real.pi 2))
    a : Int
    b : Nat
    hb : LT.lt 0 b
    h : Eq (HDiv.hDiv Real.pi 2) (HDiv.hDiv ↑a ↑b)
    ha : LT.lt 0 ↑a
    k : ∀ (n : Nat), LT.lt 0 (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) …
    j : Filter.Eventually (fun n => LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (H …
    ⊢ False
  -/
  obtain ⟨n, hn⟩ := j.exists
  /-
    case intro.intro.intro.intro
    h' : Not (Irrational (HDiv.hDiv Real.pi 2))
    a : Int
    b : Nat
    hb : LT.lt 0 b
    h : Eq (HDiv.hDiv Real.pi 2) (HDiv.hDiv ↑a ↑b)
    ha : LT.lt 0 ↑a
    k : ∀ (n : Nat), LT.lt 0 (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) …
    j : Filter.Eventually (fun n => LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (H …
    n : Nat
    hn : LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) 1) …
    ⊢ False
  -/
  have hn' : 0 < a ^ (2 * n + 1) / n ! * I n (π / 2) := mul_pos (k _) I_pos
  obtain ⟨z, hz⟩ : ∃ z : ℤ, (sinPoly n).eval₂ (Int.castRingHom ℝ) (a / b) * b ^ (2 * n + 1) = z :=
    is_integer a b ((sinPoly_natDegree_le _).trans (by omega))
  /-
    case intro.intro.intro.intro.intro
    h' : Not (Irrational (HDiv.hDiv Real.pi 2))
    a : Int
    b : Nat
    hb : LT.lt 0 b
    h : Eq (HDiv.hDiv Real.pi 2) (HDiv.hDiv ↑a ↑b)
    ha : LT.lt 0 ↑a
    k : ∀ (n : Nat), LT.lt 0 (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) …
    j : Filter.Eventually (fun n => LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (H …
    n : Nat
    hn : LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) 1) …
    hn' : LT.lt 0 (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) …
    z : Int
    hz : Eq (HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real) (HDiv.hDiv ↑a ↑b)  …
    ⊢ False
  -/
  have e := sinPoly_add_cosPoly_eval (π / 2) n
  /-
    case intro.intro.intro.intro.intro
    h' : Not (Irrational (HDiv.hDiv Real.pi 2))
    a : Int
    b : Nat
    hb : LT.lt 0 b
    h : Eq (HDiv.hDiv Real.pi 2) (HDiv.hDiv ↑a ↑b)
    ha : LT.lt 0 ↑a
    k : ∀ (n : Nat), LT.lt 0 (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) …
    j : Filter.Eventually (fun n => LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (H …
    n : Nat
    hn : LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) 1) …
    hn' : LT.lt 0 (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) …
    z : Int
    hz : Eq (HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real) (HDiv.hDiv ↑a ↑b)  …
    e : Eq (HMul.hMul (I n (HDiv.hDiv Real.pi 2)) (HPow.hPow (HDiv.hDiv Real.pi 2) …
    ⊢ False
  -/
  rw [cos_pi_div_two, sin_pi_div_two, mul_zero, mul_one, add_zero] at e
  have : a ^ (2 * n + 1) / n ! * I n (π / 2) =
      eval₂ (Int.castRingHom ℝ) (π / 2) (sinPoly n) * b ^ (2 * n + 1) := by
    nth_rw 2 [h] at e
    field_simp at e ⊢
    linear_combination e
  /-
    case intro.intro.intro.intro.intro
    h' : Not (Irrational (HDiv.hDiv Real.pi 2))
    a : Int
    b : Nat
    hb : LT.lt 0 b
    h : Eq (HDiv.hDiv Real.pi 2) (HDiv.hDiv ↑a ↑b)
    ha : LT.lt 0 ↑a
    k : ∀ (n : Nat), LT.lt 0 (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) …
    j : Filter.Eventually (fun n => LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (H …
    n : Nat
    hn : LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) 1) …
    hn' : LT.lt 0 (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) …
    z : Int
    hz : Eq (HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real) (HDiv.hDiv ↑a ↑b)  …
    e : Eq (HMul.hMul (I n (HDiv.hDiv Real.pi 2)) (HPow.hPow (HDiv.hDiv Real.pi 2) …
    this : Eq (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) 1)) …
    ⊢ False
  -/
  have : (0 : ℝ) < z ∧ (z : ℝ) < 1 := by simp [← hz, ← h, ← this, hn', hn]
  /-
    case intro.intro.intro.intro.intro
    h' : Not (Irrational (HDiv.hDiv Real.pi 2))
    a : Int
    b : Nat
    hb : LT.lt 0 b
    h : Eq (HDiv.hDiv Real.pi 2) (HDiv.hDiv ↑a ↑b)
    ha : LT.lt 0 ↑a
    k : ∀ (n : Nat), LT.lt 0 (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) …
    j : Filter.Eventually (fun n => LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (H …
    n : Nat
    hn : LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) 1) …
    hn' : LT.lt 0 (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) …
    z : Int
    hz : Eq (HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real) (HDiv.hDiv ↑a ↑b)  …
    e : Eq (HMul.hMul (I n (HDiv.hDiv Real.pi 2)) (HPow.hPow (HDiv.hDiv Real.pi 2) …
    this✝ : Eq (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) 1) …
    this : And (LT.lt 0 ↑z) (LT.lt (↑z) 1)
    ⊢ False
  -/
  norm_cast at this
  /-
    case intro.intro.intro.intro.intro
    h' : Not (Irrational (HDiv.hDiv Real.pi 2))
    a : Int
    b : Nat
    hb : LT.lt 0 b
    h : Eq (HDiv.hDiv Real.pi 2) (HDiv.hDiv ↑a ↑b)
    ha : LT.lt 0 ↑a
    k : ∀ (n : Nat), LT.lt 0 (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) …
    j : Filter.Eventually (fun n => LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (H …
    n : Nat
    hn : LT.lt (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) 1) …
    hn' : LT.lt 0 (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) …
    z : Int
    hz : Eq (HMul.hMul (Polynomial.eval₂ (Int.castRingHom Real) (HDiv.hDiv ↑a ↑b)  …
    e : Eq (HMul.hMul (I n (HDiv.hDiv Real.pi 2)) (HPow.hPow (HDiv.hDiv Real.pi 2) …
    this✝ : Eq (HMul.hMul (HDiv.hDiv (HPow.hPow (↑a) (HAdd.hAdd (HMul.hMul 2 n) 1) …
    this : And (LT.lt 0 z) (LT.lt z 1)
    ⊢ False
  -/
  omega
  /-
    🎉 no goals
  -/


