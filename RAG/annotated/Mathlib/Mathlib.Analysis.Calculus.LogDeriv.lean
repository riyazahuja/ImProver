/-- The logarithmic derivative of a function defined as `deriv f /f`. Note that it will be zero
at `x` if `f` is not DifferentiableAt `x`. -/
def logDeriv (f : 𝕜 → 𝕜') :=
  deriv f / f


theorem logDeriv_apply (f : 𝕜 → 𝕜') (x : 𝕜) : logDeriv f x = deriv f x / f x := rfl


lemma logDeriv_eq_zero_of_not_differentiableAt (f : 𝕜 → 𝕜') (x : 𝕜) (h : ¬DifferentiableAt 𝕜 f x) :
    logDeriv f x = 0 := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    f : 𝕜 → 𝕜'
    x : 𝕜
    h : Not (DifferentiableAt 𝕜 f x)
    ⊢ Eq (logDeriv f x) 0
  -/
  simp only [logDeriv_apply, deriv_zero_of_not_differentiableAt h, zero_div]
  /-
    🎉 no goals
  -/


@[simp]
theorem logDeriv_id (x : 𝕜) : logDeriv id x = 1 / x := by
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    ⊢ Eq (logDeriv id x) (HDiv.hDiv 1 x)
  -/
  simp [logDeriv_apply]
  /-
    🎉 no goals
  -/


@[simp] theorem logDeriv_id' (x : 𝕜) : logDeriv (·) x = 1 / x := logDeriv_id x


@[simp]
theorem logDeriv_const (a : 𝕜') : logDeriv (fun _ : 𝕜 ↦ a) = 0 := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    a : 𝕜'
    ⊢ Eq (logDeriv fun x => a) 0
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    a : 𝕜'
    x✝ : 𝕜
    ⊢ Eq (logDeriv (fun x => a) x✝) (0 x✝)
  -/
  simp [logDeriv_apply]
  /-
    🎉 no goals
  -/


theorem logDeriv_mul {f g : 𝕜 → 𝕜'} (x : 𝕜) (hf : f x ≠ 0) (hg : g x ≠ 0)
    (hdf : DifferentiableAt 𝕜 f x) (hdg : DifferentiableAt 𝕜 g x) :
      logDeriv (fun z => f z * g z) x = logDeriv f x + logDeriv g x := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    f g : 𝕜 → 𝕜'
    x : 𝕜
    hf : Ne (f x) 0
    hg : Ne (g x) 0
    hdf : DifferentiableAt 𝕜 f x
    hdg : DifferentiableAt 𝕜 g x
    ⊢ Eq (logDeriv (fun z => HMul.hMul (f z) (g z)) x) (HAdd.hAdd (logDeriv f x) ( …
  -/
  simp only [logDeriv_apply, deriv_mul hdf hdg]
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    f g : 𝕜 → 𝕜'
    x : 𝕜
    hf : Ne (f x) 0
    hg : Ne (g x) 0
    hdf : DifferentiableAt 𝕜 f x
    hdg : DifferentiableAt 𝕜 g x
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul (deriv f x) (g x)) (HMul.hMul (f x) (der …
  -/
  field_simp [mul_comm]
  /-
    🎉 no goals
  -/


theorem logDeriv_div {f g : 𝕜 → 𝕜'} (x : 𝕜) (hf : f x ≠ 0) (hg : g x ≠ 0)
    (hdf : DifferentiableAt 𝕜 f x) (hdg : DifferentiableAt 𝕜 g x) :
    logDeriv (fun z => f z / g z) x = logDeriv f x - logDeriv g x := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    f g : 𝕜 → 𝕜'
    x : 𝕜
    hf : Ne (f x) 0
    hg : Ne (g x) 0
    hdf : DifferentiableAt 𝕜 f x
    hdg : DifferentiableAt 𝕜 g x
    ⊢ Eq (logDeriv (fun z => HDiv.hDiv (f z) (g z)) x) (HSub.hSub (logDeriv f x) ( …
  -/
  simp only [logDeriv_apply, deriv_div hdf hdg]
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    f g : 𝕜 → 𝕜'
    x : 𝕜
    hf : Ne (f x) 0
    hg : Ne (g x) 0
    hdf : DifferentiableAt 𝕜 f x
    hdg : DifferentiableAt 𝕜 g x
    ⊢ Eq (HDiv.hDiv (deriv (fun z => HDiv.hDiv (f z) (g z)) x) (HDiv.hDiv (f x) (g …
  -/
  field_simp [mul_comm]
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    f g : 𝕜 → 𝕜'
    x : 𝕜
    hf : Ne (f x) 0
    hg : Ne (g x) 0
    hdf : DifferentiableAt 𝕜 f x
    hdg : DifferentiableAt 𝕜 g x
    ⊢ Eq (HMul.hMul (HMul.hMul (f x) (g x)) (HMul.hMul (g x) (HSub.hSub (HMul.hMul …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem logDeriv_mul_const {f : 𝕜 → 𝕜'} (x : 𝕜) (a : 𝕜') (ha : a ≠ 0):
    logDeriv (fun z => f z * a) x = logDeriv f x := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    f : 𝕜 → 𝕜'
    x : 𝕜
    a : 𝕜'
    ha : Ne a 0
    ⊢ Eq (logDeriv (fun z => HMul.hMul (f z) a) x) (logDeriv f x)
  -/
  simp only [logDeriv_apply, deriv_mul_const_field, mul_div_mul_right _ _ ha]
  /-
    🎉 no goals
  -/


theorem logDeriv_const_mul {f : 𝕜 → 𝕜'} (x : 𝕜) (a : 𝕜') (ha : a ≠ 0):
    logDeriv (fun z => a * f z) x = logDeriv f x := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    f : 𝕜 → 𝕜'
    x : 𝕜
    a : 𝕜'
    ha : Ne a 0
    ⊢ Eq (logDeriv (fun z => HMul.hMul a (f z)) x) (logDeriv f x)
  -/
  simp only [logDeriv_apply, deriv_const_mul_field, mul_div_mul_left _ _ ha]
  /-
    🎉 no goals
  -/


/-- The logarithmic derivative of a finite product is the sum of the logarithmic derivatives. -/
theorem logDeriv_prod {ι : Type*} (s : Finset ι) (f : ι → 𝕜 → 𝕜') (x : 𝕜) (hf : ∀ i ∈ s, f i x ≠ 0)
    (hd : ∀ i ∈ s, DifferentiableAt 𝕜 (f i) x) :
    logDeriv (∏ i ∈ s, f i ·) x = ∑ i ∈ s, logDeriv (f i) x := by
  induction s using Finset.cons_induction with
  | empty => simp
  | cons a s ha ih =>
    rw [Finset.forall_mem_cons] at hf hd
    simp_rw [Finset.prod_cons, Finset.sum_cons]
    rw [logDeriv_mul, ih hf.2 hd.2]
    · exact hf.1
    · simpa [Finset.prod_eq_zero_iff] using hf.2
    · exact hd.1
    · exact .finset_prod hd.2


lemma logDeriv_fun_zpow {f : 𝕜 → 𝕜'} {x : 𝕜} (hdf : DifferentiableAt 𝕜 f x) (n : ℤ) :
    logDeriv (f · ^ n) x = n * logDeriv f x := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    f : 𝕜 → 𝕜'
    x : 𝕜
    hdf : DifferentiableAt 𝕜 f x
    n : Int
    ⊢ Eq (logDeriv (fun x => HPow.hPow (f x) n) x) (HMul.hMul (↑n) (logDeriv f x))
  -/
  rcases eq_or_ne n 0 with rfl | hn; · simp
                                       /-
                                         🎉 no goals
                                       -/
  /-
    case inr
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    f : 𝕜 → 𝕜'
    x : 𝕜
    hdf : DifferentiableAt 𝕜 f x
    n : Int
    hn : Ne n 0
    ⊢ Eq (logDeriv (fun x => HPow.hPow (f x) n) x) (HMul.hMul (↑n) (logDeriv f x))
  -/
  rcases eq_or_ne (f x) 0 with hf | hf
    /-
      case inr.inl
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NontriviallyNormedField 𝕜'
      inst✝ : NormedAlgebra 𝕜 𝕜'
      f : 𝕜 → 𝕜'
      x : 𝕜
      hdf : DifferentiableAt 𝕜 f x
      n : Int
      hn : Ne n 0
      hf : Eq (f x) 0
      ⊢ Eq (logDeriv (fun x => HPow.hPow (f x) n) x) (HMul.hMul (↑n) (logDeriv f x))
    -/
  · simp [logDeriv_apply, zero_zpow, *]
    /-
      🎉 no goals
    -/
  · rw [logDeriv_apply, ← comp_def (·^n), deriv_comp _ (differentiableAt_zpow.2 <| .inl hf) hdf,
      deriv_zpow, logDeriv_apply]
    /-
      case inr.inr
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NontriviallyNormedField 𝕜'
      inst✝ : NormedAlgebra 𝕜 𝕜'
      f : 𝕜 → 𝕜'
      x : 𝕜
      hdf : DifferentiableAt 𝕜 f x
      n : Int
      hn : Ne n 0
      hf : Ne (f x) 0
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (↑n) (HPow.hPow (f x) (HSub.hSub n 1)))  …
    -/
    field_simp [zpow_ne_zero, zpow_sub_one₀ hf]
    /-
      case inr.inr
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NontriviallyNormedField 𝕜'
      inst✝ : NormedAlgebra 𝕜 𝕜'
      f : 𝕜 → 𝕜'
      x : 𝕜
      hdf : DifferentiableAt 𝕜 f x
      n : Int
      hn : Ne n 0
      hf : Ne (f x) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (↑n) (HPow.hPow (f x) n)) (deriv f x)) ( …
    -/
    ring
    /-
      🎉 no goals
    -/


lemma logDeriv_fun_pow {f : 𝕜 → 𝕜'} {x : 𝕜} (hdf : DifferentiableAt 𝕜 f x) (n : ℕ) :
    logDeriv (f · ^ n) x = n * logDeriv f x :=
  mod_cast logDeriv_fun_zpow hdf n


@[simp]
lemma logDeriv_zpow (x : 𝕜) (n : ℤ) : logDeriv (· ^ n) x = n / x := by
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    n : Int
    ⊢ Eq (logDeriv (fun x => HPow.hPow x n) x) (HDiv.hDiv (↑n) x)
  -/
  rw [logDeriv_fun_zpow (by fun_prop), logDeriv_id', mul_one_div]
  /-
    🎉 no goals
  -/


@[simp]
lemma logDeriv_pow (x : 𝕜) (n : ℕ) : logDeriv (· ^ n) x = n / x :=
  mod_cast logDeriv_zpow x n


@[simp] lemma logDeriv_inv (x : 𝕜) : logDeriv (·⁻¹) x = -1 / x := by
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    ⊢ Eq (logDeriv (fun x => Inv.inv x) x) (HDiv.hDiv (-1) x)
  -/
  simpa using logDeriv_zpow x (-1)
  /-
    🎉 no goals
  -/


theorem logDeriv_comp {f : 𝕜' → 𝕜'} {g : 𝕜 → 𝕜'} {x : 𝕜} (hf : DifferentiableAt 𝕜' f (g x))
    (hg : DifferentiableAt 𝕜 g x) : logDeriv (f ∘ g) x = logDeriv f (g x) * deriv g x := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    f : 𝕜' → 𝕜'
    g : 𝕜 → 𝕜'
    x : 𝕜
    hf : DifferentiableAt 𝕜' f (g x)
    hg : DifferentiableAt 𝕜 g x
    ⊢ Eq (logDeriv (Function.comp f g) x) (HMul.hMul (logDeriv f (g x)) (deriv g x))
  -/
  simp only [logDeriv, Pi.div_apply, deriv_comp _ hf hg, comp_apply]
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    f : 𝕜' → 𝕜'
    g : 𝕜 → 𝕜'
    x : 𝕜
    hf : DifferentiableAt 𝕜' f (g x)
    hg : DifferentiableAt 𝕜 g x
    ⊢ Eq (HDiv.hDiv (HMul.hMul (deriv f (g x)) (deriv g x)) (f (g x))) (HMul.hMul  …
  -/
  ring
  /-
    🎉 no goals
  -/

