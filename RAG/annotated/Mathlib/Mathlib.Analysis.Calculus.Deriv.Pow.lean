theorem hasStrictDerivAt_pow :
    ∀ (n : ℕ) (x : 𝕜), HasStrictDerivAt (fun x : 𝕜 ↦ x ^ n) ((n : 𝕜) * x ^ (n - 1)) x
               /-
                 𝕜 : Type u
                 inst✝ : NontriviallyNormedField 𝕜
                 x : 𝕜
                 ⊢ HasStrictDerivAt (fun x => HPow.hPow x 0) (HMul.hMul (↑0) (HPow.hPow x (HSub …
               -/
  | 0, x => by simp [hasStrictDerivAt_const]
               /-
                 🎉 no goals
               -/
               /-
                 𝕜 : Type u
                 inst✝ : NontriviallyNormedField 𝕜
                 x : 𝕜
                 ⊢ HasStrictDerivAt (fun x => HPow.hPow x 1) (HMul.hMul (↑1) (HPow.hPow x (HSub …
               -/
  | 1, x => by simpa using hasStrictDerivAt_id x
               /-
                 🎉 no goals
               -/
  | n + 1 + 1, x => by
    simpa [pow_succ, add_mul, mul_assoc] using
      (hasStrictDerivAt_pow (n + 1) x).mul (hasStrictDerivAt_id x)


theorem hasDerivAt_pow (n : ℕ) (x : 𝕜) :
    HasDerivAt (fun x : 𝕜 => x ^ n) ((n : 𝕜) * x ^ (n - 1)) x :=
  (hasStrictDerivAt_pow n x).hasDerivAt


theorem hasDerivWithinAt_pow (n : ℕ) (x : 𝕜) (s : Set 𝕜) :
    HasDerivWithinAt (fun x : 𝕜 => x ^ n) ((n : 𝕜) * x ^ (n - 1)) s x :=
  (hasDerivAt_pow n x).hasDerivWithinAt


theorem differentiableAt_pow : DifferentiableAt 𝕜 (fun x : 𝕜 => x ^ n) x :=
  (hasDerivAt_pow n x).differentiableAt


theorem differentiableWithinAt_pow :
    DifferentiableWithinAt 𝕜 (fun x : 𝕜 => x ^ n) s x :=
  (differentiableAt_pow n).differentiableWithinAt


theorem differentiable_pow : Differentiable 𝕜 fun x : 𝕜 => x ^ n := fun _ => differentiableAt_pow n


theorem differentiableOn_pow : DifferentiableOn 𝕜 (fun x : 𝕜 => x ^ n) s :=
  (differentiable_pow n).differentiableOn


theorem deriv_pow : deriv (fun x : 𝕜 => x ^ n) x = (n : 𝕜) * x ^ (n - 1) :=
  (hasDerivAt_pow n x).deriv


@[simp]
theorem deriv_pow' : (deriv fun x : 𝕜 => x ^ n) = fun x => (n : 𝕜) * x ^ (n - 1) :=
  funext fun _ => deriv_pow n


theorem derivWithin_pow (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (fun x : 𝕜 => x ^ n) s x = (n : 𝕜) * x ^ (n - 1) :=
  (hasDerivWithinAt_pow n x s).derivWithin hxs


theorem HasDerivWithinAt.pow (hc : HasDerivWithinAt c c' s x) :
    HasDerivWithinAt (fun y => c y ^ n) ((n : 𝕜) * c x ^ (n - 1) * c') s x :=
  (hasDerivAt_pow n (c x)).comp_hasDerivWithinAt x hc


theorem HasDerivAt.pow (hc : HasDerivAt c c' x) :
    HasDerivAt (fun y => c y ^ n) ((n : 𝕜) * c x ^ (n - 1) * c') x := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    c : 𝕜 → 𝕜
    c' : 𝕜
    n : Nat
    hc : HasDerivAt c c' x
    ⊢ HasDerivAt (fun y => HPow.hPow (c y) n) (HMul.hMul (HMul.hMul (↑n) (HPow.hPo …
  -/
  rw [← hasDerivWithinAt_univ] at *
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    c : 𝕜 → 𝕜
    c' : 𝕜
    n : Nat
    hc : HasDerivWithinAt c c' Set.univ x
    ⊢ HasDerivWithinAt (fun y => HPow.hPow (c y) n) (HMul.hMul (HMul.hMul (↑n) (HP …
  -/
  exact hc.pow n
  /-
    🎉 no goals
  -/


theorem derivWithin_pow' (hc : DifferentiableWithinAt 𝕜 c s x) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (fun x => c x ^ n) s x = (n : 𝕜) * c x ^ (n - 1) * derivWithin c s x :=
  (hc.hasDerivWithinAt.pow n).derivWithin hxs


@[simp]
theorem deriv_pow'' (hc : DifferentiableAt 𝕜 c x) :
    deriv (fun x => c x ^ n) x = (n : 𝕜) * c x ^ (n - 1) * deriv c x :=
  (hc.hasDerivAt.pow n).deriv

