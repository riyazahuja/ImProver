/-- Formal power series over a coefficient type `R` -/
abbrev PowerSeries (R : Type*) :=
  MvPowerSeries Unit R


/--
`R⟦X⟧` is notation for `PowerSeries R`,
the semiring of formal power series in one variable over a semiring `R`.
-/
scoped notation:9000 R "⟦X⟧" => PowerSeries R


instance [Inhabited R] : Inhabited R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝ : Inhabited R
    ⊢ Inhabited (PowerSeries R)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    inst✝ : Inhabited R
    ⊢ Inhabited (MvPowerSeries Unit R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [Zero R] : Zero R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝ : Zero R
    ⊢ Zero (PowerSeries R)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    inst✝ : Zero R
    ⊢ Zero (MvPowerSeries Unit R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [AddMonoid R] : AddMonoid R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝ : AddMonoid R
    ⊢ AddMonoid (PowerSeries R)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    inst✝ : AddMonoid R
    ⊢ AddMonoid (MvPowerSeries Unit R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [AddGroup R] : AddGroup R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝ : AddGroup R
    ⊢ AddGroup (PowerSeries R)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    inst✝ : AddGroup R
    ⊢ AddGroup (MvPowerSeries Unit R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [AddCommMonoid R] : AddCommMonoid R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝ : AddCommMonoid R
    ⊢ AddCommMonoid (PowerSeries R)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    inst✝ : AddCommMonoid R
    ⊢ AddCommMonoid (MvPowerSeries Unit R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [AddCommGroup R] : AddCommGroup R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝ : AddCommGroup R
    ⊢ AddCommGroup (PowerSeries R)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    inst✝ : AddCommGroup R
    ⊢ AddCommGroup (MvPowerSeries Unit R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [Semiring R] : Semiring R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Semiring (PowerSeries R)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Semiring (MvPowerSeries Unit R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [CommSemiring R] : CommSemiring R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    ⊢ CommSemiring (PowerSeries R)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    ⊢ CommSemiring (MvPowerSeries Unit R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [Ring R] : Ring R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝ : Ring R
    ⊢ Ring (PowerSeries R)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    inst✝ : Ring R
    ⊢ Ring (MvPowerSeries Unit R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [CommRing R] : CommRing R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ CommRing (PowerSeries R)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ CommRing (MvPowerSeries Unit R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [Nontrivial R] : Nontrivial R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝ : Nontrivial R
    ⊢ Nontrivial (PowerSeries R)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    inst✝ : Nontrivial R
    ⊢ Nontrivial (MvPowerSeries Unit R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {A} [Semiring R] [AddCommMonoid A] [Module R A] : Module R A⟦X⟧ := by
  /-
    R : Type u_1
    A : Type ?u.2145
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid A
    inst✝ : Module R A
    ⊢ Module R (PowerSeries A)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    A : Type ?u.2145
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid A
    inst✝ : Module R A
    ⊢ Module R (MvPowerSeries Unit A)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {A S} [Semiring R] [Semiring S] [AddCommMonoid A] [Module R A] [Module S A] [SMul R S]
    [IsScalarTower R S A] : IsScalarTower R S A⟦X⟧ :=
  Pi.isScalarTower


instance {A} [Semiring A] [CommSemiring R] [Algebra R A] : Algebra R A⟦X⟧ := by
  /-
    R : Type u_1
    A : Type ?u.5125
    inst✝² : Semiring A
    inst✝¹ : CommSemiring R
    inst✝ : Algebra R A
    ⊢ Algebra R (PowerSeries A)
  -/
  dsimp only [PowerSeries]
  /-
    R : Type u_1
    A : Type ?u.5125
    inst✝² : Semiring A
    inst✝¹ : CommSemiring R
    inst✝ : Algebra R A
    ⊢ Algebra R (MvPowerSeries Unit A)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The `n`th coefficient of a formal power series. -/
def coeff (n : ℕ) : R⟦X⟧ →ₗ[R] R :=
  MvPowerSeries.coeff R (single () n)


/-- The `n`th monomial with coefficient `a` as formal power series. -/
def monomial (n : ℕ) : R →ₗ[R] R⟦X⟧ :=
  MvPowerSeries.monomial R (single () n)


theorem coeff_def {s : Unit →₀ ℕ} {n : ℕ} (h : s () = n) : coeff R n = MvPowerSeries.coeff R s := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    s : Finsupp Unit Nat
    n : Nat
    h : Eq (s Unit.unit) n
    ⊢ Eq (PowerSeries.coeff R n) (MvPowerSeries.coeff R s)
  -/
  rw [coeff, ← h, ← Finsupp.unique_single s]
  /-
    🎉 no goals
  -/


/-- Two formal power series are equal if all their coefficients are equal. -/
@[ext]
theorem ext {φ ψ : R⟦X⟧} (h : ∀ n, coeff R n φ = coeff R n ψ) : φ = ψ :=
  MvPowerSeries.ext fun n => by
    /-
      R : Type u_1
      inst✝ : Semiring R
      φ ψ : PowerSeries R
      h : ∀ (n : Nat), Eq ((PowerSeries.coeff R n) φ) ((PowerSeries.coeff R n) ψ)
      n : Finsupp Unit Nat
      ⊢ Eq ((MvPowerSeries.coeff R n) φ) ((MvPowerSeries.coeff R n) ψ)
    -/
    rw [← coeff_def]
      /-
        R : Type u_1
        inst✝ : Semiring R
        φ ψ : PowerSeries R
        h : ∀ (n : Nat), Eq ((PowerSeries.coeff R n) φ) ((PowerSeries.coeff R n) ψ)
        n : Finsupp Unit Nat
        ⊢ Eq ((PowerSeries.coeff R ?m.7564) φ) ((PowerSeries.coeff R ?m.7564) ψ)
      -/
    · apply h
      /-
        🎉 no goals
      -/
    /-
      R : Type u_1
      inst✝ : Semiring R
      φ ψ : PowerSeries R
      h : ∀ (n : Nat), Eq ((PowerSeries.coeff R n) φ) ((PowerSeries.coeff R n) ψ)
      n : Finsupp Unit Nat
      ⊢ Eq (n Unit.unit) ?m.7564
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem forall_coeff_eq_zero (φ : R⟦X⟧) : (∀ n, coeff R n φ = 0) ↔ φ = 0 :=
                               /-
                                 R : Type u_1
                                 inst✝ : Semiring R
                                 φ : PowerSeries R
                                 h : Eq φ 0
                                 ⊢ ∀ (n : Nat), Eq ((PowerSeries.coeff R n) φ) 0
                               -/
  ⟨fun h => ext h, fun h => by simp [h]⟩
                               /-
                                 🎉 no goals
                               -/


instance [Subsingleton R] : Subsingleton R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Subsingleton R
    ⊢ Subsingleton (PowerSeries R)
  -/
  simp only [subsingleton_iff, PowerSeries.ext_iff]
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Subsingleton R
    ⊢ ∀ (x y : PowerSeries R) (n : Nat), Eq ((PowerSeries.coeff R n) x) ((PowerSer …
  -/
  subsingleton
  /-
    🎉 no goals
  -/


/-- Constructor for formal power series. -/
def mk {R} (f : ℕ → R) : R⟦X⟧ := fun s => f (s ())


@[simp]
theorem coeff_mk (n : ℕ) (f : ℕ → R) : coeff R n (mk f) = f n :=
  congr_arg f Finsupp.single_eq_same


theorem coeff_monomial (m n : ℕ) (a : R) : coeff R m (monomial R n a) = if m = n then a else 0 :=
  calc
    coeff R m (monomial R n a) = _ := MvPowerSeries.coeff_monomial _ _ _
                                     /-
                                       R : Type u_1
                                       inst✝ : Semiring R
                                       m n : Nat
                                       a : R
                                       ⊢ Eq (ite (Eq (Finsupp.single Unit.unit m) (Finsupp.single Unit.unit n)) a 0)  …
                                     -/
    _ = if m = n then a else 0 := by simp only [Finsupp.unique_single_eq_iff]
                                     /-
                                       🎉 no goals
                                     -/


theorem monomial_eq_mk (n : ℕ) (a : R) : monomial R n a = mk fun m => if m = n then a else 0 :=
                  /-
                    R : Type u_1
                    inst✝ : Semiring R
                    n : Nat
                    a : R
                    m : Nat
                    ⊢ Eq ((PowerSeries.coeff R m) ((PowerSeries.monomial R n) a)) ((PowerSeries.co …
                  -/
  ext fun m => by rw [coeff_monomial, coeff_mk]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem coeff_monomial_same (n : ℕ) (a : R) : coeff R n (monomial R n a) = a :=
  MvPowerSeries.coeff_monomial_same _ _


@[simp]
theorem coeff_comp_monomial (n : ℕ) : (coeff R n).comp (monomial R n) = LinearMap.id :=
  LinearMap.ext <| coeff_monomial_same n


/-- The constant coefficient of a formal power series. -/
def constantCoeff : R⟦X⟧ →+* R :=
  MvPowerSeries.constantCoeff Unit R


/-- The constant formal power series. -/
def C : R →+* R⟦X⟧ :=
  MvPowerSeries.C Unit R


/-- The variable of the formal power series ring. -/
def X : R⟦X⟧ :=
  MvPowerSeries.X ()


theorem commute_X (φ : R⟦X⟧) : Commute φ X :=
  MvPowerSeries.commute_X _ _


@[simp]
theorem coeff_zero_eq_constantCoeff : ⇑(coeff R 0) = constantCoeff R := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Eq ⇑(PowerSeries.coeff R 0) ⇑(PowerSeries.constantCoeff R)
  -/
  rw [coeff, Finsupp.single_zero]
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Eq ⇑(MvPowerSeries.coeff R 0) ⇑(PowerSeries.constantCoeff R)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem coeff_zero_eq_constantCoeff_apply (φ : R⟦X⟧) : coeff R 0 φ = constantCoeff R φ := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    ⊢ Eq ((PowerSeries.coeff R 0) φ) ((PowerSeries.constantCoeff R) φ)
  -/
  rw [coeff_zero_eq_constantCoeff]
  /-
    🎉 no goals
  -/


@[simp]
theorem monomial_zero_eq_C : ⇑(monomial R 0) = C R := by
  -- This used to be `rw`, but we need `rw; rfl` after https://github.com/leanprover/lean4/pull/2644
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Eq ⇑(PowerSeries.monomial R 0) ⇑(PowerSeries.C R)
  -/
  rw [monomial, Finsupp.single_zero, MvPowerSeries.monomial_zero_eq_C]
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Eq ⇑(MvPowerSeries.C Unit R) ⇑(PowerSeries.C R)
  -/
  rfl
  /-
    🎉 no goals
  -/


                                                                        /-
                                                                          R : Type u_1
                                                                          inst✝ : Semiring R
                                                                          a : R
                                                                          ⊢ Eq ((PowerSeries.monomial R 0) a) ((PowerSeries.C R) a)
                                                                        -/
theorem monomial_zero_eq_C_apply (a : R) : monomial R 0 a = C R a := by simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem coeff_C (n : ℕ) (a : R) : coeff R n (C R a : R⟦X⟧) = if n = 0 then a else 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    a : R
    ⊢ Eq ((PowerSeries.coeff R n) ((PowerSeries.C R) a)) (ite (Eq n 0) a 0)
  -/
  rw [← monomial_zero_eq_C_apply, coeff_monomial]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_zero_C (a : R) : coeff R 0 (C R a) = a := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    a : R
    ⊢ Eq ((PowerSeries.coeff R 0) ((PowerSeries.C R) a)) a
  -/
  rw [coeff_C, if_pos rfl]
  /-
    🎉 no goals
  -/


theorem coeff_ne_zero_C {a : R} {n : ℕ} (h : n ≠ 0) : coeff R n (C R a) = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    a : R
    n : Nat
    h : Ne n 0
    ⊢ Eq ((PowerSeries.coeff R n) ((PowerSeries.C R) a)) 0
  -/
  rw [coeff_C, if_neg h]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_succ_C {a : R} {n : ℕ} : coeff R (n + 1) (C R a) = 0 :=
  coeff_ne_zero_C n.succ_ne_zero


theorem C_injective : Function.Injective (C R) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Function.Injective ⇑(PowerSeries.C R)
  -/
  intro a b H
  /-
    R : Type u_1
    inst✝ : Semiring R
    a b : R
    H : Eq ((PowerSeries.C R) a) ((PowerSeries.C R) b)
    ⊢ Eq a b
  -/
  simp_rw [PowerSeries.ext_iff] at H
  /-
    R : Type u_1
    inst✝ : Semiring R
    a b : R
    H : ∀ (n : Nat), Eq ((PowerSeries.coeff R n) ((PowerSeries.C R) a)) ((PowerSer …
    ⊢ Eq a b
  -/
  simpa only [coeff_zero_C] using H 0
  /-
    🎉 no goals
  -/


protected theorem subsingleton_iff : Subsingleton R⟦X⟧ ↔ Subsingleton R := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Iff (Subsingleton (PowerSeries R)) (Subsingleton R)
  -/
  refine ⟨fun h ↦ ?_, fun _ ↦ inferInstance⟩
  /-
    R : Type u_1
    inst✝ : Semiring R
    h : Subsingleton (PowerSeries R)
    ⊢ Subsingleton R
  -/
  rw [subsingleton_iff] at h ⊢
  /-
    R : Type u_1
    inst✝ : Semiring R
    h : ∀ (x y : PowerSeries R), Eq x y
    ⊢ ∀ (x y : R), Eq x y
  -/
  exact fun a b ↦ C_injective (h (C R a) (C R b))
  /-
    🎉 no goals
  -/


theorem X_eq : (X : R⟦X⟧) = monomial R 1 1 :=
  rfl


theorem coeff_X (n : ℕ) : coeff R n (X : R⟦X⟧) = if n = 1 then 1 else 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq ((PowerSeries.coeff R n) PowerSeries.X) (ite (Eq n 1) 1 0)
  -/
  rw [X_eq, coeff_monomial]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_zero_X : coeff R 0 (X : R⟦X⟧) = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Eq ((PowerSeries.coeff R 0) PowerSeries.X) 0
  -/
  rw [coeff, Finsupp.single_zero, X, MvPowerSeries.coeff_zero_X]
  /-
    🎉 no goals
  -/


@[simp]
                                                     /-
                                                       R : Type u_1
                                                       inst✝ : Semiring R
                                                       ⊢ Eq ((PowerSeries.coeff R 1) PowerSeries.X) 1
                                                     -/
theorem coeff_one_X : coeff R 1 (X : R⟦X⟧) = 1 := by rw [coeff_X, if_pos rfl]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem X_ne_zero [Nontrivial R] : (X : R⟦X⟧) ≠ 0 := fun H => by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    H : Eq PowerSeries.X 0
    ⊢ False
  -/
  simpa only [coeff_one_X, one_ne_zero, map_zero] using congr_arg (coeff R 1) H
  /-
    🎉 no goals
  -/


theorem X_pow_eq (n : ℕ) : (X : R⟦X⟧) ^ n = monomial R n 1 :=
  MvPowerSeries.X_pow_eq _ n


theorem coeff_X_pow (m n : ℕ) : coeff R m ((X : R⟦X⟧) ^ n) = if m = n then 1 else 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    m n : Nat
    ⊢ Eq ((PowerSeries.coeff R m) (HPow.hPow PowerSeries.X n)) (ite (Eq m n) 1 0)
  -/
  rw [X_pow_eq, coeff_monomial]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_X_pow_self (n : ℕ) : coeff R n ((X : R⟦X⟧) ^ n) = 1 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq ((PowerSeries.coeff R n) (HPow.hPow PowerSeries.X n)) 1
  -/
  rw [coeff_X_pow, if_pos rfl]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_one (n : ℕ) : coeff R n (1 : R⟦X⟧) = if n = 0 then 1 else 0 :=
  coeff_C n 1


theorem coeff_zero_one : coeff R 0 (1 : R⟦X⟧) = 1 :=
  coeff_zero_C 1


theorem coeff_mul (n : ℕ) (φ ψ : R⟦X⟧) :
    coeff R n (φ * ψ) = ∑ p ∈ antidiagonal n, coeff R p.1 φ * coeff R p.2 ψ := by
  -- `rw` can't see that `PowerSeries = MvPowerSeries Unit`, so use `.trans`
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    φ ψ : PowerSeries R
    ⊢ Eq ((PowerSeries.coeff R n) (HMul.hMul φ ψ)) ((Finset.HasAntidiagonal.antidi …
  -/
  refine (MvPowerSeries.coeff_mul _ φ ψ).trans ?_
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    φ ψ : PowerSeries R
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (Finsupp.single Unit.unit n)).sum f …
  -/
  rw [Finsupp.antidiagonal_single, Finset.sum_map]
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    φ ψ : PowerSeries R
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun x => HMul.hMul ((MvPower …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_mul_C (n : ℕ) (φ : R⟦X⟧) (a : R) : coeff R n (φ * C R a) = coeff R n φ * a :=
  MvPowerSeries.coeff_mul_C _ φ a


@[simp]
theorem coeff_C_mul (n : ℕ) (φ : R⟦X⟧) (a : R) : coeff R n (C R a * φ) = a * coeff R n φ :=
  MvPowerSeries.coeff_C_mul _ φ a


@[simp]
theorem coeff_smul {S : Type*} [Semiring S] [Module R S] (n : ℕ) (φ : PowerSeries S) (a : R) :
    coeff S n (a • φ) = a • coeff S n φ :=
  rfl


@[simp]
theorem constantCoeff_smul {S : Type*} [Semiring S] [Module R S] (φ : PowerSeries S) (a : R) :
    constantCoeff S (a • φ) = a • constantCoeff S φ :=
  rfl


theorem smul_eq_C_mul (f : R⟦X⟧) (a : R) : a • f = C R a * f := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : PowerSeries R
    a : R
    ⊢ Eq (HSMul.hSMul a f) (HMul.hMul ((PowerSeries.C R) a) f)
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    f : PowerSeries R
    a : R
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff R n✝) (HSMul.hSMul a f)) ((PowerSeries.coeff R n✝) (H …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_succ_mul_X (n : ℕ) (φ : R⟦X⟧) : coeff R (n + 1) (φ * X) = coeff R n φ := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    φ : PowerSeries R
    ⊢ Eq ((PowerSeries.coeff R (HAdd.hAdd n 1)) (HMul.hMul φ PowerSeries.X)) ((Pow …
  -/
  simp only [coeff, Finsupp.single_add]
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    φ : PowerSeries R
    ⊢ Eq ((MvPowerSeries.coeff R (HAdd.hAdd (Finsupp.single Unit.unit n) (Finsupp. …
  -/
  convert φ.coeff_add_mul_monomial (single () n) (single () 1) _
  /-
    case h.e'_3
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    φ : PowerSeries R
    ⊢ Eq ((MvPowerSeries.coeff R (Finsupp.single Unit.unit n)) φ) (HMul.hMul ((MvP …
  -/
  rw [mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_succ_X_mul (n : ℕ) (φ : R⟦X⟧) : coeff R (n + 1) (X * φ) = coeff R n φ := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    φ : PowerSeries R
    ⊢ Eq ((PowerSeries.coeff R (HAdd.hAdd n 1)) (HMul.hMul PowerSeries.X φ)) ((Pow …
  -/
  simp only [coeff, Finsupp.single_add, add_comm n 1]
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    φ : PowerSeries R
    ⊢ Eq ((MvPowerSeries.coeff R (HAdd.hAdd (Finsupp.single Unit.unit 1) (Finsupp. …
  -/
  convert φ.coeff_add_monomial_mul (single () 1) (single () n) _
  /-
    case h.e'_3
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    φ : PowerSeries R
    ⊢ Eq ((MvPowerSeries.coeff R (Finsupp.single Unit.unit n)) φ) (HMul.hMul 1 ((M …
  -/
  rw [one_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_C (a : R) : constantCoeff R (C R a) = a :=
  rfl


@[simp]
theorem constantCoeff_comp_C : (constantCoeff R).comp (C R) = RingHom.id R :=
  rfl


@[simp]
theorem constantCoeff_zero : constantCoeff R 0 = 0 :=
  rfl


@[simp]
theorem constantCoeff_one : constantCoeff R 1 = 1 :=
  rfl


@[simp]
theorem constantCoeff_X : constantCoeff R X = 0 :=
  MvPowerSeries.coeff_zero_X _


@[simp]
theorem constantCoeff_mk {f : ℕ → R} : constantCoeff R (mk f) = f 0 := rfl


                                                                  /-
                                                                    R : Type u_1
                                                                    inst✝ : Semiring R
                                                                    φ : PowerSeries R
                                                                    ⊢ Eq ((PowerSeries.coeff R 0) (HMul.hMul φ PowerSeries.X)) 0
                                                                  -/
theorem coeff_zero_mul_X (φ : R⟦X⟧) : coeff R 0 (φ * X) = 0 := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                                                  /-
                                                                    R : Type u_1
                                                                    inst✝ : Semiring R
                                                                    φ : PowerSeries R
                                                                    ⊢ Eq ((PowerSeries.coeff R 0) (HMul.hMul PowerSeries.X φ)) 0
                                                                  -/
theorem coeff_zero_X_mul (φ : R⟦X⟧) : coeff R 0 (X * φ) = 0 := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem constantCoeff_surj : Function.Surjective (constantCoeff R) :=
  fun r => ⟨(C R) r, constantCoeff_C r⟩

-- The following section duplicates the API of `Data.Polynomial.Coeff` and should attempt to keep
-- up to date with that

theorem coeff_C_mul_X_pow (x : R) (k n : ℕ) :
    coeff R n (C R x * X ^ k : R⟦X⟧) = if n = k then x else 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x : R
    k n : Nat
    ⊢ Eq ((PowerSeries.coeff R n) (HMul.hMul ((PowerSeries.C R) x) (HPow.hPow Powe …
  -/
  simp [X_pow_eq, coeff_monomial]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_mul_X_pow (p : R⟦X⟧) (n d : ℕ) :
    coeff R (d + n) (p * X ^ n) = coeff R d p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : PowerSeries R
    n d : Nat
    ⊢ Eq ((PowerSeries.coeff R (HAdd.hAdd d n)) (HMul.hMul p (HPow.hPow PowerSerie …
  -/
  rw [coeff_mul, Finset.sum_eq_single (d, n), coeff_X_pow, if_pos rfl, mul_one]
    /-
      case h₀
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      ⊢ ∀ (b : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal (H …
    -/
  · rintro ⟨i, j⟩ h1 h2
    /-
      case h₀.mk
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d i j : Nat
      h1 : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d n)) { fs …
      h2 : Ne { fst := i, snd := j } { fst := d, snd := n }
      ⊢ Eq (HMul.hMul ((PowerSeries.coeff R { fst := i, snd := j }.1) p) ((PowerSeri …
    -/
    rw [coeff_X_pow, if_neg, mul_zero]
    /-
      case h₀.mk.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d i j : Nat
      h1 : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d n)) { fs …
      h2 : Ne { fst := i, snd := j } { fst := d, snd := n }
      ⊢ Not (Eq { fst := i, snd := j }.2 n)
    -/
    rintro rfl
    /-
      case h₀.mk.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      d i j : Nat
      h1 : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d { fst := …
      h2 : Ne { fst := i, snd := j } { fst := d, snd := { fst := i, snd := j }.2 }
      ⊢ False
    -/
    apply h2
    /-
      case h₀.mk.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      d i j : Nat
      h1 : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d { fst := …
      h2 : Ne { fst := i, snd := j } { fst := d, snd := { fst := i, snd := j }.2 }
      ⊢ Eq { fst := i, snd := j } { fst := d, snd := { fst := i, snd := j }.2 }
    -/
    rw [mem_antidiagonal, add_right_cancel_iff] at h1
    /-
      case h₀.mk.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      d i j : Nat
      h1 : Eq { fst := i, snd := j }.1 d
      h2 : Ne { fst := i, snd := j } { fst := d, snd := { fst := i, snd := j }.2 }
      ⊢ Eq { fst := i, snd := j } { fst := d, snd := { fst := i, snd := j }.2 }
    -/
    subst h1
    /-
      case h₀.mk.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      i j : Nat
      h2 : Ne { fst := i, snd := j } { fst := { fst := i, snd := j }.1, snd := { fst …
      ⊢ Eq { fst := i, snd := j } { fst := { fst := i, snd := j }.1, snd := { fst := …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h₁
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      ⊢ Not (Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d n)) {  …
    -/
  · exact fun h1 => (h1 (mem_antidiagonal.2 rfl)).elim
    /-
      🎉 no goals
    -/


@[simp]
theorem coeff_X_pow_mul (p : R⟦X⟧) (n d : ℕ) :
    coeff R (d + n) (X ^ n * p) = coeff R d p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : PowerSeries R
    n d : Nat
    ⊢ Eq ((PowerSeries.coeff R (HAdd.hAdd d n)) (HMul.hMul (HPow.hPow PowerSeries. …
  -/
  rw [coeff_mul, Finset.sum_eq_single (n, d), coeff_X_pow, if_pos rfl, one_mul]
    /-
      case h₀
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      ⊢ ∀ (b : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal (H …
    -/
  · rintro ⟨i, j⟩ h1 h2
    /-
      case h₀.mk
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d i j : Nat
      h1 : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d n)) { fs …
      h2 : Ne { fst := i, snd := j } { fst := n, snd := d }
      ⊢ Eq (HMul.hMul ((PowerSeries.coeff R { fst := i, snd := j }.1) (HPow.hPow Pow …
    -/
    rw [coeff_X_pow, if_neg, zero_mul]
    /-
      case h₀.mk.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d i j : Nat
      h1 : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d n)) { fs …
      h2 : Ne { fst := i, snd := j } { fst := n, snd := d }
      ⊢ Not (Eq { fst := i, snd := j }.1 n)
    -/
    rintro rfl
    /-
      case h₀.mk.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      d i j : Nat
      h1 : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d { fst := …
      h2 : Ne { fst := i, snd := j } { fst := { fst := i, snd := j }.1, snd := d }
      ⊢ False
    -/
    apply h2
    /-
      case h₀.mk.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      d i j : Nat
      h1 : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d { fst := …
      h2 : Ne { fst := i, snd := j } { fst := { fst := i, snd := j }.1, snd := d }
      ⊢ Eq { fst := i, snd := j } { fst := { fst := i, snd := j }.1, snd := d }
    -/
    rw [mem_antidiagonal, add_comm, add_right_cancel_iff] at h1
    /-
      case h₀.mk.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      d i j : Nat
      h1 : Eq { fst := i, snd := j }.2 d
      h2 : Ne { fst := i, snd := j } { fst := { fst := i, snd := j }.1, snd := d }
      ⊢ Eq { fst := i, snd := j } { fst := { fst := i, snd := j }.1, snd := d }
    -/
    subst h1
    /-
      case h₀.mk.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      i j : Nat
      h2 : Ne { fst := i, snd := j } { fst := { fst := i, snd := j }.1, snd := { fst …
      ⊢ Eq { fst := i, snd := j } { fst := { fst := i, snd := j }.1, snd := { fst := …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h₁
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      ⊢ Not (Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d n)) {  …
    -/
  · rw [add_comm]
    /-
      case h₁
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      ⊢ Not (Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n d)) {  …
    -/
    exact fun h1 => (h1 (mem_antidiagonal.2 rfl)).elim
    /-
      🎉 no goals
    -/


theorem coeff_mul_X_pow' (p : R⟦X⟧) (n d : ℕ) :
    coeff R d (p * X ^ n) = ite (n ≤ d) (coeff R (d - n) p) 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : PowerSeries R
    n d : Nat
    ⊢ Eq ((PowerSeries.coeff R d) (HMul.hMul p (HPow.hPow PowerSeries.X n))) (ite  …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      h : LE.le n d
      ⊢ Eq ((PowerSeries.coeff R d) (HMul.hMul p (HPow.hPow PowerSeries.X n))) ((Pow …
    -/
  · rw [← tsub_add_cancel_of_le h, coeff_mul_X_pow, add_tsub_cancel_right]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      h : Not (LE.le n d)
      ⊢ Eq ((PowerSeries.coeff R d) (HMul.hMul p (HPow.hPow PowerSeries.X n))) 0
    -/
  · refine (coeff_mul _ _ _).trans (Finset.sum_eq_zero fun x hx => ?_)
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      h : Not (LE.le n d)
      x : Prod Nat Nat
      hx : Membership.mem (Finset.HasAntidiagonal.antidiagonal d) x
      ⊢ Eq (HMul.hMul ((PowerSeries.coeff R x.1) p) ((PowerSeries.coeff R x.2) (HPow …
    -/
    rw [coeff_X_pow, if_neg, mul_zero]
    /-
      case neg.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      h : Not (LE.le n d)
      x : Prod Nat Nat
      hx : Membership.mem (Finset.HasAntidiagonal.antidiagonal d) x
      ⊢ Not (Eq x.2 n)
    -/
    exact ((le_of_add_le_right (mem_antidiagonal.mp hx).le).trans_lt <| not_le.mp h).ne
    /-
      🎉 no goals
    -/


theorem coeff_X_pow_mul' (p : R⟦X⟧) (n d : ℕ) :
    coeff R d (X ^ n * p) = ite (n ≤ d) (coeff R (d - n) p) 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : PowerSeries R
    n d : Nat
    ⊢ Eq ((PowerSeries.coeff R d) (HMul.hMul (HPow.hPow PowerSeries.X n) p)) (ite  …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      h : LE.le n d
      ⊢ Eq ((PowerSeries.coeff R d) (HMul.hMul (HPow.hPow PowerSeries.X n) p)) ((Pow …
    -/
  · rw [← tsub_add_cancel_of_le h, coeff_X_pow_mul]
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      h : LE.le n d
      ⊢ Eq ((PowerSeries.coeff R (HSub.hSub d n)) p) ((PowerSeries.coeff R (HSub.hSu …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      h : Not (LE.le n d)
      ⊢ Eq ((PowerSeries.coeff R d) (HMul.hMul (HPow.hPow PowerSeries.X n) p)) 0
    -/
  · refine (coeff_mul _ _ _).trans (Finset.sum_eq_zero fun x hx => ?_)
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      h : Not (LE.le n d)
      x : Prod Nat Nat
      hx : Membership.mem (Finset.HasAntidiagonal.antidiagonal d) x
      ⊢ Eq (HMul.hMul ((PowerSeries.coeff R x.1) (HPow.hPow PowerSeries.X n)) ((Powe …
    -/
    rw [coeff_X_pow, if_neg, zero_mul]
    /-
      case neg.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      h : Not (LE.le n d)
      x : Prod Nat Nat
      hx : Membership.mem (Finset.HasAntidiagonal.antidiagonal d) x
      ⊢ Not (Eq x.1 n)
    -/
    have := mem_antidiagonal.mp hx
    /-
      case neg.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      h : Not (LE.le n d)
      x : Prod Nat Nat
      hx : Membership.mem (Finset.HasAntidiagonal.antidiagonal d) x
      this : Eq (HAdd.hAdd x.1 x.2) d
      ⊢ Not (Eq x.1 n)
    -/
    rw [add_comm] at this
    /-
      case neg.hnc
      R : Type u_1
      inst✝ : Semiring R
      p : PowerSeries R
      n d : Nat
      h : Not (LE.le n d)
      x : Prod Nat Nat
      hx : Membership.mem (Finset.HasAntidiagonal.antidiagonal d) x
      this : Eq (HAdd.hAdd x.2 x.1) d
      ⊢ Not (Eq x.1 n)
    -/
    exact ((le_of_add_le_right this.le).trans_lt <| not_le.mp h).ne
    /-
      🎉 no goals
    -/


/-- If a formal power series is invertible, then so is its constant coefficient. -/
theorem isUnit_constantCoeff (φ : R⟦X⟧) (h : IsUnit φ) : IsUnit (constantCoeff R φ) :=
  MvPowerSeries.isUnit_constantCoeff φ h


/-- Split off the constant coefficient. -/
theorem eq_shift_mul_X_add_const (φ : R⟦X⟧) :
    φ = (mk fun p => coeff R (p + 1) φ) * X + C R (constantCoeff R φ) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    ⊢ Eq φ (HAdd.hAdd (HMul.hMul (PowerSeries.mk fun p => (PowerSeries.coeff R (HA …
  -/
  ext (_ | n)
  · simp only [coeff_zero_eq_constantCoeff, map_add, map_mul, constantCoeff_X,
      mul_zero, coeff_zero_C, zero_add]
  · simp only [coeff_succ_mul_X, coeff_mk, LinearMap.map_add, coeff_C, n.succ_ne_zero, sub_zero,
      if_false, add_zero]


/-- Split off the constant coefficient. -/
theorem eq_X_mul_shift_add_const (φ : R⟦X⟧) :
    φ = (X * mk fun p => coeff R (p + 1) φ) + C R (constantCoeff R φ) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    ⊢ Eq φ (HAdd.hAdd (HMul.hMul PowerSeries.X (PowerSeries.mk fun p => (PowerSeri …
  -/
  ext (_ | n)
  · simp only [coeff_zero_eq_constantCoeff, map_add, map_mul, constantCoeff_X,
      zero_mul, coeff_zero_C, zero_add]
  · simp only [coeff_succ_X_mul, coeff_mk, LinearMap.map_add, coeff_C, n.succ_ne_zero, sub_zero,
      if_false, add_zero]


/-- The map between formal power series induced by a map on the coefficients. -/
def map : R⟦X⟧ →+* S⟦X⟧ :=
  MvPowerSeries.map _ f


@[simp]
theorem map_id : (map (RingHom.id R) : R⟦X⟧ → R⟦X⟧) = id :=
  rfl


theorem map_comp : map (g.comp f) = (map g).comp (map f) :=
  rfl


@[simp]
theorem coeff_map (n : ℕ) (φ : R⟦X⟧) : coeff S n (map f φ) = f (coeff R n φ) :=
  rfl


@[simp]
theorem map_C (r : R) : map f (C _ r) = C _ (f r) := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    S : Type u_2
    inst✝ : Semiring S
    f : RingHom R S
    r : R
    ⊢ Eq ((PowerSeries.map f) ((PowerSeries.C R) r)) ((PowerSeries.C S) (f r))
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝¹ : Semiring R
    S : Type u_2
    inst✝ : Semiring S
    f : RingHom R S
    r : R
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff S n✝) ((PowerSeries.map f) ((PowerSeries.C R) r))) (( …
  -/
  simp [coeff_C, apply_ite f]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_X : map f X = X := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    S : Type u_2
    inst✝ : Semiring S
    f : RingHom R S
    ⊢ Eq ((PowerSeries.map f) PowerSeries.X) PowerSeries.X
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝¹ : Semiring R
    S : Type u_2
    inst✝ : Semiring S
    f : RingHom R S
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff S n✝) ((PowerSeries.map f) PowerSeries.X)) ((PowerSer …
  -/
  simp [coeff_X, apply_ite f]
  /-
    🎉 no goals
  -/


theorem X_pow_dvd_iff {n : ℕ} {φ : R⟦X⟧} :
    (X : R⟦X⟧) ^ n ∣ φ ↔ ∀ m, m < n → coeff R m φ = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    φ : PowerSeries R
    ⊢ Iff (Dvd.dvd (HPow.hPow PowerSeries.X n) φ) (∀ (m : Nat), LT.lt m n → Eq ((P …
  -/
  convert@MvPowerSeries.X_pow_dvd_iff Unit R _ () n φ
  /-
    case h.e'_2.a
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    φ : PowerSeries R
    ⊢ Iff (∀ (m : Nat), LT.lt m n → Eq ((PowerSeries.coeff R m) φ) 0) (∀ (m : Fins …
  -/
  constructor <;> intro h m hm
    /-
      case h.e'_2.a.mp
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      φ : PowerSeries R
      h : ∀ (m : Nat), LT.lt m n → Eq ((PowerSeries.coeff R m) φ) 0
      m : Finsupp Unit Nat
      hm : LT.lt (m Unit.unit) n
      ⊢ Eq ((MvPowerSeries.coeff R m) φ) 0
    -/
  · rw [Finsupp.unique_single m]
    /-
      case h.e'_2.a.mp
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      φ : PowerSeries R
      h : ∀ (m : Nat), LT.lt m n → Eq ((PowerSeries.coeff R m) φ) 0
      m : Finsupp Unit Nat
      hm : LT.lt (m Unit.unit) n
      ⊢ Eq ((MvPowerSeries.coeff R (Finsupp.single Inhabited.default (m Inhabited.de …
    -/
    convert h _ hm
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.a.mpr
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      φ : PowerSeries R
      h : ∀ (m : Finsupp Unit Nat), LT.lt (m Unit.unit) n → Eq ((MvPowerSeries.coeff …
      m : Nat
      hm : LT.lt m n
      ⊢ Eq ((PowerSeries.coeff R m) φ) 0
    -/
  · apply h
    /-
      case h.e'_2.a.mpr.a
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      φ : PowerSeries R
      h : ∀ (m : Finsupp Unit Nat), LT.lt (m Unit.unit) n → Eq ((MvPowerSeries.coeff …
      m : Nat
      hm : LT.lt m n
      ⊢ LT.lt ((Finsupp.single Unit.unit m) Unit.unit) n
    -/
    simpa only [Finsupp.single_eq_same] using hm
    /-
      🎉 no goals
    -/


theorem X_dvd_iff {φ : R⟦X⟧} : (X : R⟦X⟧) ∣ φ ↔ constantCoeff R φ = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    ⊢ Iff (Dvd.dvd PowerSeries.X φ) (Eq ((PowerSeries.constantCoeff R) φ) 0)
  -/
  rw [← pow_one (X : R⟦X⟧), X_pow_dvd_iff, ← coeff_zero_eq_constantCoeff_apply]
  /-
    R : Type u_1
    inst✝ : Semiring R
    φ : PowerSeries R
    ⊢ Iff (∀ (m : Nat), LT.lt m 1 → Eq ((PowerSeries.coeff R m) φ) 0) (Eq ((PowerS …
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      inst✝ : Semiring R
      φ : PowerSeries R
      h : ∀ (m : Nat), LT.lt m 1 → Eq ((PowerSeries.coeff R m) φ) 0
      ⊢ Eq ((PowerSeries.coeff R 0) φ) 0
    -/
  · exact h 0 zero_lt_one
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝ : Semiring R
      φ : PowerSeries R
      h : Eq ((PowerSeries.coeff R 0) φ) 0
      ⊢ ∀ (m : Nat), LT.lt m 1 → Eq ((PowerSeries.coeff R m) φ) 0
    -/
  · intro m hm
    /-
      case mpr
      R : Type u_1
      inst✝ : Semiring R
      φ : PowerSeries R
      h : Eq ((PowerSeries.coeff R 0) φ) 0
      m : Nat
      hm : LT.lt m 1
      ⊢ Eq ((PowerSeries.coeff R m) φ) 0
    -/
    rwa [Nat.eq_zero_of_le_zero (Nat.le_of_succ_le_succ hm)]
    /-
      🎉 no goals
    -/


/-- The ring homomorphism taking a power series `f(X)` to `f(aX)`. -/
noncomputable def rescale (a : R) : R⟦X⟧ →+* R⟦X⟧ where
  toFun f := PowerSeries.mk fun n => a ^ n * PowerSeries.coeff R n f
  map_zero' := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      ⊢ Eq ((↑{ toFun := fun f => PowerSeries.mk fun n => HMul.hMul (HPow.hPow a n)  …
    -/
    ext
    /-
      case h
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      n✝ : Nat
      ⊢ Eq ((PowerSeries.coeff R n✝) ((↑{ toFun := fun f => PowerSeries.mk fun n =>  …
    -/
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      ⊢ Eq ((fun f => PowerSeries.mk fun n => HMul.hMul (HPow.hPow a n) ((PowerSerie …
    -/
    simp only [LinearMap.map_zero, PowerSeries.coeff_mk, mul_zero]
    /-
      case h
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      n✝ : Nat
      ⊢ Eq ((PowerSeries.coeff R n✝) ((fun f => PowerSeries.mk fun n => HMul.hMul (H …
    -/
    /-
      🎉 no goals
    -/
    /-
      case h
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      n✝ : Nat
      ⊢ Eq (ite (Eq n✝ 0) (HPow.hPow a n✝) 0) (ite (Eq n✝ 0) 1 0)
    -/
  map_one' := by
      /-
        case pos
        R : Type u_1
        inst✝ : CommSemiring R
        a : R
        n✝ : Nat
        h : Eq n✝ 0
        ⊢ Eq (HPow.hPow a n✝) 1
      -/
    ext1
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      n✝ : Nat
      h : Not (Eq n✝ 0)
      ⊢ Eq 0 0
    -/
    simp only [mul_boole, PowerSeries.coeff_mk, PowerSeries.coeff_one]
    /-
      🎉 no goals
    -/
    split_ifs with h
    · rw [h, pow_zero a]
    rfl
  map_add' := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      ⊢ ∀ (x y : PowerSeries R), Eq ((↑{ toFun := fun f => PowerSeries.mk fun n => H …
    -/
    intros
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      f g : PowerSeries R
      ⊢ Eq ({ toFun := fun f => PowerSeries.mk fun n => HMul.hMul (HPow.hPow a n) (( …
    -/
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      x✝ y✝ : PowerSeries R
      ⊢ Eq ((↑{ toFun := fun f => PowerSeries.mk fun n => HMul.hMul (HPow.hPow a n)  …
    -/
    /-
      case h
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      f g : PowerSeries R
      n✝ : Nat
      ⊢ Eq ((PowerSeries.coeff R n✝) ({ toFun := fun f => PowerSeries.mk fun n => HM …
    -/
    ext
    /-
      case h
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      f g : PowerSeries R
      n✝ : Nat
      ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n✝).sum fun i => HMul.hMul (HPow.hP …
    -/
    /-
      case h
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      x✝ y✝ : PowerSeries R
      n✝ : Nat
      ⊢ Eq ((PowerSeries.coeff R n✝) ((↑{ toFun := fun f => PowerSeries.mk fun n =>  …
    -/
    /-
      case h
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      f g : PowerSeries R
      n✝ : Nat
      ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal n✝ …
    -/
    dsimp only
    /-
      case h
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      f g : PowerSeries R
      n✝ : Nat
      ⊢ ∀ (a_1 b : Nat), Eq (HAdd.hAdd a_1 b) n✝ → Eq (HMul.hMul (HPow.hPow a n✝) (H …
    -/
    /-
      case h
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      x✝ y✝ : PowerSeries R
      n✝ : Nat
      ⊢ Eq ((PowerSeries.coeff R n✝) (PowerSeries.mk fun n => HMul.hMul (HPow.hPow a …
    -/
    /-
      case h
      R : Type u_1
      inst✝ : CommSemiring R
      a : R
      f g : PowerSeries R
      n✝ b c : Nat
      H : Eq (HAdd.hAdd b c) n✝
      ⊢ Eq (HMul.hMul (HPow.hPow a n✝) (HMul.hMul ((PowerSeries.coeff R b) f) ((Powe …
    -/
    exact mul_add _ _ _
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  map_mul' f g := by
    ext
    rw [PowerSeries.coeff_mul, PowerSeries.coeff_mk, PowerSeries.coeff_mul, Finset.mul_sum]
    apply sum_congr rfl
    simp only [coeff_mk, Prod.forall, mem_antidiagonal]
    intro b c H
    rw [← H, pow_add, mul_mul_mul_comm]


@[simp]
theorem coeff_rescale (f : R⟦X⟧) (a : R) (n : ℕ) :
    coeff R n (rescale a f) = a ^ n * coeff R n f :=
  coeff_mk n (fun n ↦ a ^ n * (coeff R n) f)


@[simp]
theorem rescale_zero : rescale 0 = (C R).comp (constantCoeff R) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq (PowerSeries.rescale 0) ((PowerSeries.C R).comp (PowerSeries.constantCoef …
  -/
  ext x n
  simp only [Function.comp_apply, RingHom.coe_comp, rescale, RingHom.coe_mk,
    PowerSeries.coeff_mk _ _, coeff_C]
  /-
    case a.h
    R : Type u_1
    inst✝ : CommSemiring R
    x : PowerSeries R
    n : Nat
    ⊢ Eq ((PowerSeries.coeff R n) ({ toFun := fun f => PowerSeries.mk fun n => HMu …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


                                                                         /-
                                                                           R : Type u_1
                                                                           inst✝ : CommSemiring R
                                                                           ⊢ Eq ((PowerSeries.rescale 0) PowerSeries.X) ((PowerSeries.C R) ((PowerSeries. …
                                                                         -/
theorem rescale_zero_apply : rescale 0 X = C R (constantCoeff R X) := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
theorem rescale_one : rescale 1 = RingHom.id R⟦X⟧ := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq (PowerSeries.rescale 1) (RingHom.id (PowerSeries R))
  -/
  ext
  /-
    case a.h
    R : Type u_1
    inst✝ : CommSemiring R
    x✝ : PowerSeries R
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff R n✝) ((PowerSeries.rescale 1) x✝)) ((PowerSeries.coe …
  -/
  simp only [coeff_rescale, one_pow, one_mul, RingHom.id_apply]
  /-
    🎉 no goals
  -/


theorem rescale_mk (f : ℕ → R) (a : R) : rescale a (mk f) = mk fun n : ℕ => a ^ n * f n := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f : Nat → R
    a : R
    ⊢ Eq ((PowerSeries.rescale a) (PowerSeries.mk f)) (PowerSeries.mk fun n => HMu …
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    f : Nat → R
    a : R
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff R n✝) ((PowerSeries.rescale a) (PowerSeries.mk f))) ( …
  -/
  rw [coeff_rescale, coeff_mk, coeff_mk]
  /-
    🎉 no goals
  -/


theorem rescale_rescale (f : R⟦X⟧) (a b : R) :
    rescale b (rescale a f) = rescale (a * b) f := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f : PowerSeries R
    a b : R
    ⊢ Eq ((PowerSeries.rescale b) ((PowerSeries.rescale a) f)) ((PowerSeries.resca …
  -/
  ext n
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    f : PowerSeries R
    a b : R
    n : Nat
    ⊢ Eq ((PowerSeries.coeff R n) ((PowerSeries.rescale b) ((PowerSeries.rescale a …
  -/
  simp_rw [coeff_rescale]
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    f : PowerSeries R
    a b : R
    n : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow b n) (HMul.hMul (HPow.hPow a n) ((PowerSeries.coeff …
  -/
  rw [mul_pow, mul_comm _ (b ^ n), mul_assoc]
  /-
    🎉 no goals
  -/


theorem rescale_mul (a b : R) : rescale (a * b) = (rescale b).comp (rescale a) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    a b : R
    ⊢ Eq (PowerSeries.rescale (HMul.hMul a b)) ((PowerSeries.rescale b).comp (Powe …
  -/
  ext
  /-
    case a.h
    R : Type u_1
    inst✝ : CommSemiring R
    a b : R
    x✝ : PowerSeries R
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff R n✝) ((PowerSeries.rescale (HMul.hMul a b)) x✝)) ((P …
  -/
  simp [← rescale_rescale]
  /-
    🎉 no goals
  -/


/-- Coefficients of a product of power series -/
theorem coeff_prod (f : ι → PowerSeries R) (d : ℕ) (s : Finset ι) :
    coeff R d (∏ j ∈ s, f j) = ∑ l ∈ finsuppAntidiag s d, ∏ i ∈ s, coeff R (l i) (f i) := by
  /-
    R : Type u_2
    inst✝¹ : CommSemiring R
    ι : Type u_3
    inst✝ : DecidableEq ι
    f : ι → PowerSeries R
    d : Nat
    s : Finset ι
    ⊢ Eq ((PowerSeries.coeff R d) (s.prod fun j => f j)) ((s.finsuppAntidiag d).su …
  -/
  simp only [coeff]
  rw [MvPowerSeries.coeff_prod, ← AddEquiv.finsuppUnique_symm d, ← mapRange_finsuppAntidiag_eq,
    sum_map, sum_congr rfl]
  /-
    R : Type u_2
    inst✝¹ : CommSemiring R
    ι : Type u_3
    inst✝ : DecidableEq ι
    f : ι → PowerSeries R
    d : Nat
    s : Finset ι
    ⊢ ∀ (x : Finsupp ι Nat), Membership.mem (s.finsuppAntidiag d) x → Eq (s.prod f …
  -/
  intro x _
  /-
    R : Type u_2
    inst✝¹ : CommSemiring R
    ι : Type u_3
    inst✝ : DecidableEq ι
    f : ι → PowerSeries R
    d : Nat
    s : Finset ι
    x : Finsupp ι Nat
    a✝ : Membership.mem (s.finsuppAntidiag d) x
    ⊢ Eq (s.prod fun i => (MvPowerSeries.coeff R (((Finsupp.mapRange.addEquiv AddE …
  -/
  apply prod_congr rfl
  /-
    R : Type u_2
    inst✝¹ : CommSemiring R
    ι : Type u_3
    inst✝ : DecidableEq ι
    f : ι → PowerSeries R
    d : Nat
    s : Finset ι
    x : Finsupp ι Nat
    a✝ : Membership.mem (s.finsuppAntidiag d) x
    ⊢ ∀ (x_1 : ι), Membership.mem s x_1 → Eq ((MvPowerSeries.coeff R (((Finsupp.ma …
  -/
  intro i _
  /-
    R : Type u_2
    inst✝¹ : CommSemiring R
    ι : Type u_3
    inst✝ : DecidableEq ι
    f : ι → PowerSeries R
    d : Nat
    s : Finset ι
    x : Finsupp ι Nat
    a✝¹ : Membership.mem (s.finsuppAntidiag d) x
    i : ι
    a✝ : Membership.mem s i
    ⊢ Eq ((MvPowerSeries.coeff R (((Finsupp.mapRange.addEquiv AddEquiv.finsuppUniq …
  -/
  congr 2
  simp only [AddEquiv.toEquiv_eq_coe, Finsupp.mapRange.addEquiv_toEquiv, AddEquiv.toEquiv_symm,
    Equiv.coe_toEmbedding, Finsupp.mapRange.equiv_apply, AddEquiv.coe_toEquiv_symm,
    Finsupp.mapRange_apply, AddEquiv.finsuppUnique_symm]


/-- The `n`-th coefficient of the `k`-th power of a power series. -/
lemma coeff_pow (k n : ℕ) (φ : R⟦X⟧) :
    coeff R n (φ ^ k) = ∑ l ∈ finsuppAntidiag (range k) n, ∏ i ∈ range k, coeff R (l i) φ := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    k n : Nat
    φ : PowerSeries R
    ⊢ Eq ((PowerSeries.coeff R n) (HPow.hPow φ k)) (((Finset.range k).finsuppAntid …
  -/
  have h₁ (i : ℕ) : Function.const ℕ φ i = φ := rfl
  have h₂ (i : ℕ) : ∏ j ∈ range i, Function.const ℕ φ j = φ ^ i := by
    apply prod_range_induction (fun _ => φ) (fun i => φ ^ i) rfl (congrFun rfl) i
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    k n : Nat
    φ : PowerSeries R
    h₁ : ∀ (i : Nat), Eq (Function.const Nat φ i) φ
    h₂ : ∀ (i : Nat), Eq ((Finset.range i).prod fun j => Function.const Nat φ j) ( …
    ⊢ Eq ((PowerSeries.coeff R n) (HPow.hPow φ k)) (((Finset.range k).finsuppAntid …
  -/
  rw [← h₂, ← h₁ k]
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    k n : Nat
    φ : PowerSeries R
    h₁ : ∀ (i : Nat), Eq (Function.const Nat φ i) φ
    h₂ : ∀ (i : Nat), Eq ((Finset.range i).prod fun j => Function.const Nat φ j) ( …
    ⊢ Eq ((PowerSeries.coeff R n) ((Finset.range k).prod fun j => Function.const N …
  -/
  apply coeff_prod (f := Function.const ℕ φ) (d := n) (s := range k)
  /-
    🎉 no goals
  -/


/-- First coefficient of the product of two power series. -/
lemma coeff_one_mul (φ ψ : R⟦X⟧) : coeff R 1 (φ * ψ) =
    coeff R 1 φ * constantCoeff R ψ + coeff R 1 ψ * constantCoeff R φ := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    φ ψ : PowerSeries R
    ⊢ Eq ((PowerSeries.coeff R 1) (HMul.hMul φ ψ)) (HAdd.hAdd (HMul.hMul ((PowerSe …
  -/
  have : Finset.antidiagonal 1 = {(0, 1), (1, 0)} := by exact rfl
  rw [coeff_mul, this, Finset.sum_insert, Finset.sum_singleton, coeff_zero_eq_constantCoeff,
    mul_comm, add_comm]
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    φ ψ : PowerSeries R
    this : Eq (Finset.HasAntidiagonal.antidiagonal 1) (Insert.insert { fst := 0, s …
    ⊢ Not (Membership.mem (Singleton.singleton { fst := 1, snd := 0 }) { fst := 0, …
  -/
  norm_num
  /-
    🎉 no goals
  -/


/-- First coefficient of the `n`-th power of a power series. -/
lemma coeff_one_pow (n : ℕ) (φ : R⟦X⟧) :
    coeff R 1 (φ ^ n) = n * coeff R 1 φ * (constantCoeff R φ) ^ (n - 1) := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    n : Nat
    φ : PowerSeries R
    ⊢ Eq ((PowerSeries.coeff R 1) (HPow.hPow φ n)) (HMul.hMul (HMul.hMul (↑n) ((Po …
  -/
  rcases Nat.eq_zero_or_pos n with (rfl | hn)
    /-
      case inl
      R : Type u_2
      inst✝ : CommSemiring R
      φ : PowerSeries R
      ⊢ Eq ((PowerSeries.coeff R 1) (HPow.hPow φ 0)) (HMul.hMul (HMul.hMul (↑0) ((Po …
    -/
  · simp
    /-
      🎉 no goals
    -/
  induction n with
  | zero => omega
  | succ n' ih =>
      have h₁ (m : ℕ) : φ ^ (m + 1) = φ ^ m * φ := by exact rfl
      have h₂ : Finset.antidiagonal 1 = {(0, 1), (1, 0)} := by exact rfl
      rw [h₁, coeff_mul, h₂, Finset.sum_insert, Finset.sum_singleton]
      · simp only [coeff_zero_eq_constantCoeff, map_pow, Nat.cast_add, Nat.cast_one,
          add_tsub_cancel_right]
        have h₀ : n' = 0 ∨ 1 ≤ n' := by omega
        rcases h₀ with h' | h'
        · by_contra h''
          rw [h'] at h''
          simp only [pow_zero, one_mul, coeff_one, one_ne_zero, ↓reduceIte, zero_mul, add_zero,
            CharP.cast_eq_zero, zero_add, mul_one, not_true_eq_false] at h''
          norm_num at h''
        · rw [ih]
          · conv => lhs; arg 2; rw [mul_comm, ← mul_assoc]
            move_mul [← (constantCoeff R) φ ^ (n' - 1)]
            conv => enter [1, 2, 1, 1, 2]; rw [← pow_one (a := constantCoeff R φ)]
            rw [← pow_add (a := constantCoeff R φ)]
            conv => enter [1, 2, 1, 1]; rw [Nat.sub_add_cancel h']
            conv => enter [1, 2, 1]; rw [mul_comm]
            rw [mul_assoc, ← one_add_mul, add_comm, mul_assoc]
            conv => enter [1, 2]; rw [mul_comm]
          exact h'
      · decide


theorem not_isField : ¬IsField A⟦X⟧ := by
  /-
    A : Type u_2
    inst✝ : CommRing A
    ⊢ Not (IsField (PowerSeries A))
  -/
  by_cases hA : Subsingleton A
    /-
      case pos
      A : Type u_2
      inst✝ : CommRing A
      hA : Subsingleton A
      ⊢ Not (IsField (PowerSeries A))
    -/
  · exact not_isField_of_subsingleton _
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_2
      inst✝ : CommRing A
      hA : Not (Subsingleton A)
      ⊢ Not (IsField (PowerSeries A))
    -/
  · nontriviality A
    /-
      A : Type u_2
      inst✝ : CommRing A
      hA : Not (Subsingleton A)
      a✝ : Nontrivial A
      ⊢ Not (IsField (PowerSeries A))
    -/
    rw [Ring.not_isField_iff_exists_ideal_bot_lt_and_lt_top]
    /-
      A : Type u_2
      inst✝ : CommRing A
      hA : Not (Subsingleton A)
      a✝ : Nontrivial A
      ⊢ Exists fun I => And (LT.lt Bot.bot I) (LT.lt I Top.top)
    -/
    use Ideal.span {X}
    /-
      case h
      A : Type u_2
      inst✝ : CommRing A
      hA : Not (Subsingleton A)
      a✝ : Nontrivial A
      ⊢ And (LT.lt Bot.bot (Ideal.span (Singleton.singleton PowerSeries.X))) (LT.lt  …
    -/
    constructor
      /-
        case h.left
        A : Type u_2
        inst✝ : CommRing A
        hA : Not (Subsingleton A)
        a✝ : Nontrivial A
        ⊢ LT.lt Bot.bot (Ideal.span (Singleton.singleton PowerSeries.X))
      -/
    · rw [bot_lt_iff_ne_bot, Ne, Ideal.span_singleton_eq_bot]
      /-
        case h.left
        A : Type u_2
        inst✝ : CommRing A
        hA : Not (Subsingleton A)
        a✝ : Nontrivial A
        ⊢ Not (Eq PowerSeries.X 0)
      -/
      exact X_ne_zero
      /-
        🎉 no goals
      -/
    · rw [lt_top_iff_ne_top, Ne, Ideal.eq_top_iff_one, Ideal.mem_span_singleton,
        X_dvd_iff, constantCoeff_one]
      /-
        case h.right
        A : Type u_2
        inst✝ : CommRing A
        hA : Not (Subsingleton A)
        a✝ : Nontrivial A
        ⊢ Not (Eq 1 0)
      -/
      exact one_ne_zero
      /-
        🎉 no goals
      -/


@[simp]
theorem rescale_X (a : A) : rescale a X = C A a * X := by
  /-
    A : Type u_2
    inst✝ : CommRing A
    a : A
    ⊢ Eq ((PowerSeries.rescale a) PowerSeries.X) (HMul.hMul ((PowerSeries.C A) a)  …
  -/
  ext
  /-
    case h
    A : Type u_2
    inst✝ : CommRing A
    a : A
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff A n✝) ((PowerSeries.rescale a) PowerSeries.X)) ((Powe …
  -/
  simp only [coeff_rescale, coeff_C_mul, coeff_X]
  /-
    case h
    A : Type u_2
    inst✝ : CommRing A
    a : A
    n✝ : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow a n✝) (ite (Eq n✝ 1) 1 0)) (HMul.hMul a (ite (Eq n✝ …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


theorem rescale_neg_one_X : rescale (-1 : A) X = -X := by
  /-
    A : Type u_2
    inst✝ : CommRing A
    ⊢ Eq ((PowerSeries.rescale (-1)) PowerSeries.X) (Neg.neg PowerSeries.X)
  -/
  rw [rescale_X, map_neg, map_one, neg_one_mul]
  /-
    🎉 no goals
  -/


/-- The ring homomorphism taking a power series `f(X)` to `f(-X)`. -/
noncomputable def evalNegHom : A⟦X⟧ →+* A⟦X⟧ :=
  rescale (-1 : A)


@[simp]
theorem evalNegHom_X : evalNegHom (X : A⟦X⟧) = -X :=
  rescale_neg_one_X


theorem eq_zero_or_eq_zero_of_mul_eq_zero [NoZeroDivisors R] (φ ψ : R⟦X⟧) (h : φ * ψ = 0) :
    φ = 0 ∨ ψ = 0 := by
  classical
  rw [or_iff_not_imp_left]
  intro H
  have ex : ∃ m, coeff R m φ ≠ 0 := by
    contrapose! H
    exact ext H
  let m := Nat.find ex
  have hm₁ : coeff R m φ ≠ 0 := Nat.find_spec ex
  have hm₂ : ∀ k < m, ¬coeff R k φ ≠ 0 := fun k => Nat.find_min ex
  ext n
  rw [(coeff R n).map_zero]
  induction' n using Nat.strong_induction_on with n ih
  replace h := congr_arg (coeff R (m + n)) h
  rw [LinearMap.map_zero, coeff_mul, Finset.sum_eq_single (m, n)] at h
  · replace h := NoZeroDivisors.eq_zero_or_eq_zero_of_mul_eq_zero h
    rw [or_iff_not_imp_left] at h
    exact h hm₁
  · rintro ⟨i, j⟩ hij hne
    by_cases hj : j < n
    · rw [ih j hj, mul_zero]
    by_cases hi : i < m
    · specialize hm₂ _ hi
      push_neg at hm₂
      rw [hm₂, zero_mul]
    rw [mem_antidiagonal] at hij
    push_neg at hi hj
    suffices m < i by
      have : m + n < i + j := add_lt_add_of_lt_of_le this hj
      exfalso
      exact ne_of_lt this hij.symm
    contrapose! hne
    obtain rfl := le_antisymm hi hne
    simpa [Ne, Prod.mk.inj_iff] using (add_right_inj m).mp hij
  · contrapose!
    intro
    rw [mem_antidiagonal]


instance [NoZeroDivisors R] : NoZeroDivisors R⟦X⟧ where
  eq_zero_or_eq_zero_of_mul_eq_zero := eq_zero_or_eq_zero_of_mul_eq_zero _ _


instance [IsDomain R] : IsDomain R⟦X⟧ :=
  NoZeroDivisors.to_isDomain _


/-- The ideal spanned by the variable in the power series ring
 over an integral domain is a prime ideal. -/
theorem span_X_isPrime : (Ideal.span ({X} : Set R⟦X⟧)).IsPrime := by
  suffices Ideal.span ({X} : Set R⟦X⟧) = RingHom.ker (constantCoeff R) by
    rw [this]
    exact RingHom.ker_isPrime _
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ Eq (Ideal.span (Singleton.singleton PowerSeries.X)) (RingHom.ker (PowerSerie …
  -/
  apply Ideal.ext
  /-
    case h
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ ∀ (x : PowerSeries R), Iff (Membership.mem (Ideal.span (Singleton.singleton  …
  -/
  intro φ
  /-
    case h
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    φ : PowerSeries R
    ⊢ Iff (Membership.mem (Ideal.span (Singleton.singleton PowerSeries.X)) φ) (Mem …
  -/
  rw [RingHom.mem_ker, Ideal.mem_span_singleton, X_dvd_iff]
  /-
    🎉 no goals
  -/


/-- The variable of the power series ring over an integral domain is prime. -/
theorem X_prime : Prime (X : R⟦X⟧) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ Prime PowerSeries.X
  -/
  rw [← Ideal.span_singleton_prime]
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ⊢ (Ideal.span (Singleton.singleton PowerSeries.X)).IsPrime
    -/
  · exact span_X_isPrime
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ⊢ Ne PowerSeries.X 0
    -/
  · intro h
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      h : Eq PowerSeries.X 0
      ⊢ False
    -/
    simpa [map_zero (coeff R 1)] using congr_arg (coeff R 1) h
    /-
      🎉 no goals
    -/


/-- The variable of the power series ring over an integral domain is irreducible. -/
theorem X_irreducible : Irreducible (X : R⟦X⟧) := X_prime.irreducible


theorem rescale_injective {a : R} (ha : a ≠ 0) : Function.Injective (rescale a) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    a : R
    ha : Ne a 0
    ⊢ Function.Injective ⇑(PowerSeries.rescale a)
  -/
  intro p q h
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    a : R
    ha : Ne a 0
    p q : PowerSeries R
    h : Eq ((PowerSeries.rescale a) p) ((PowerSeries.rescale a) q)
    ⊢ Eq p q
  -/
  rw [PowerSeries.ext_iff] at *
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    a : R
    ha : Ne a 0
    p q : PowerSeries R
    h : ∀ (n : Nat), Eq ((PowerSeries.coeff R n) ((PowerSeries.rescale a) p)) ((Po …
    ⊢ ∀ (n : Nat), Eq ((PowerSeries.coeff R n) p) ((PowerSeries.coeff R n) q)
  -/
  intro n
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    a : R
    ha : Ne a 0
    p q : PowerSeries R
    h : ∀ (n : Nat), Eq ((PowerSeries.coeff R n) ((PowerSeries.rescale a) p)) ((Po …
    n : Nat
    ⊢ Eq ((PowerSeries.coeff R n) p) ((PowerSeries.coeff R n) q)
  -/
  specialize h n
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    a : R
    ha : Ne a 0
    p q : PowerSeries R
    n : Nat
    h : Eq ((PowerSeries.coeff R n) ((PowerSeries.rescale a) p)) ((PowerSeries.coe …
    ⊢ Eq ((PowerSeries.coeff R n) p) ((PowerSeries.coeff R n) q)
  -/
  rw [coeff_rescale, coeff_rescale, mul_eq_mul_left_iff] at h
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    a : R
    ha : Ne a 0
    p q : PowerSeries R
    n : Nat
    h : Or (Eq ((PowerSeries.coeff R n) p) ((PowerSeries.coeff R n) q)) (Eq (HPow. …
    ⊢ Eq ((PowerSeries.coeff R n) p) ((PowerSeries.coeff R n) q)
  -/
  apply h.resolve_right
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    a : R
    ha : Ne a 0
    p q : PowerSeries R
    n : Nat
    h : Or (Eq ((PowerSeries.coeff R n) p) ((PowerSeries.coeff R n) q)) (Eq (HPow. …
    ⊢ Not (Eq (HPow.hPow a n) 0)
  -/
  intro h'
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    a : R
    ha : Ne a 0
    p q : PowerSeries R
    n : Nat
    h : Or (Eq ((PowerSeries.coeff R n) p) ((PowerSeries.coeff R n) q)) (Eq (HPow. …
    h' : Eq (HPow.hPow a n) 0
    ⊢ False
  -/
  exact ha (pow_eq_zero h')
  /-
    🎉 no goals
  -/


theorem C_eq_algebraMap {r : R} : C R r = (algebraMap R R⟦X⟧) r :=
  rfl


theorem algebraMap_apply {r : R} : algebraMap R A⟦X⟧ r = C A (algebraMap R A r) :=
  MvPowerSeries.algebraMap_apply


instance [Nontrivial R] : Nontrivial (Subalgebra R R⟦X⟧) :=
  { inferInstanceAs <| Nontrivial <| Subalgebra R <| MvPowerSeries Unit R with }


/-- The natural inclusion from polynomials into formal power series. -/
@[coe]
def toPowerSeries : R[X] → (PowerSeries R) := fun φ =>
  PowerSeries.mk fun n => coeff φ n


@[deprecated (since := "2024-10-27")] alias ToPowerSeries := toPowerSeries


/-- The natural inclusion from polynomials into formal power series. -/
instance coeToPowerSeries : Coe R[X] (PowerSeries R) :=
  ⟨toPowerSeries⟩


theorem coe_def : (φ : PowerSeries R) = PowerSeries.mk (coeff φ) :=
  rfl


@[simp, norm_cast]
theorem coeff_coe (n) : PowerSeries.coeff R n φ = coeff φ n :=
  congr_arg (coeff φ) Finsupp.single_eq_same


@[simp, norm_cast]
theorem coe_monomial (n : ℕ) (a : R) :
    (monomial n a : PowerSeries R) = PowerSeries.monomial R n a := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    n : Nat
    a : R
    ⊢ Eq (↑((Polynomial.monomial n) a)) ((PowerSeries.monomial R n) a)
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    n : Nat
    a : R
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff R n✝) ↑((Polynomial.monomial n) a)) ((PowerSeries.coe …
  -/
  simp [coeff_coe, PowerSeries.coeff_monomial, Polynomial.coeff_monomial, eq_comm]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_zero : ((0 : R[X]) : PowerSeries R) = 0 :=
  rfl


@[simp, norm_cast]
theorem coe_one : ((1 : R[X]) : PowerSeries R) = 1 := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq (↑1) 1
  -/
  have := coe_monomial 0 (1 : R)
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    this : Eq (↑((Polynomial.monomial 0) 1)) ((PowerSeries.monomial R 0) 1)
    ⊢ Eq (↑1) 1
  -/
  rwa [PowerSeries.monomial_zero_eq_C_apply] at this
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_add : ((φ + ψ : R[X]) : PowerSeries R) = φ + ψ := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    φ ψ : Polynomial R
    ⊢ Eq (↑(HAdd.hAdd φ ψ)) (HAdd.hAdd ↑φ ↑ψ)
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    φ ψ : Polynomial R
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff R n✝) ↑(HAdd.hAdd φ ψ)) ((PowerSeries.coeff R n✝) (HA …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_mul : ((φ * ψ : R[X]) : PowerSeries R) = φ * ψ :=
                              /-
                                R : Type u_1
                                inst✝ : CommSemiring R
                                φ ψ : Polynomial R
                                n : Nat
                                ⊢ Eq ((PowerSeries.coeff R n) ↑(HMul.hMul φ ψ)) ((PowerSeries.coeff R n) (HMul …
                              -/
  PowerSeries.ext fun n => by simp only [coeff_coe, PowerSeries.coeff_mul, coeff_mul]
                              /-
                                🎉 no goals
                              -/


@[simp, norm_cast]
theorem coe_C (a : R) : ((C a : R[X]) : PowerSeries R) = PowerSeries.C R a := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    a : R
    ⊢ Eq (↑(Polynomial.C a)) ((PowerSeries.C R) a)
  -/
  have := coe_monomial 0 a
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    a : R
    this : Eq (↑((Polynomial.monomial 0) a)) ((PowerSeries.monomial R 0) a)
    ⊢ Eq (↑(Polynomial.C a)) ((PowerSeries.C R) a)
  -/
  rwa [PowerSeries.monomial_zero_eq_C_apply] at this
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_X : ((X : R[X]) : PowerSeries R) = PowerSeries.X :=
  coe_monomial _ _


@[simp]
theorem constantCoeff_coe : PowerSeries.constantCoeff R φ = φ.coeff 0 :=
  rfl


theorem coe_injective : Function.Injective (Coe.coe : R[X] → PowerSeries R) := fun x y h => by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    x y : Polynomial R
    h : Eq (Coe.coe x) (Coe.coe y)
    ⊢ Eq x y
  -/
  ext
  /-
    case a
    R : Type u_1
    inst✝ : CommSemiring R
    x y : Polynomial R
    h : Eq (Coe.coe x) (Coe.coe y)
    n✝ : Nat
    ⊢ Eq (x.coeff n✝) (y.coeff n✝)
  -/
  simp_rw [← coeff_coe]
  /-
    case a
    R : Type u_1
    inst✝ : CommSemiring R
    x y : Polynomial R
    h : Eq (Coe.coe x) (Coe.coe y)
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff R n✝) ↑x) ((PowerSeries.coeff R n✝) ↑y)
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_inj : (φ : PowerSeries R) = ψ ↔ φ = ψ :=
  (coe_injective R).eq_iff


@[simp]
                                                                /-
                                                                  R : Type u_1
                                                                  inst✝ : CommSemiring R
                                                                  φ : Polynomial R
                                                                  ⊢ Iff (Eq (↑φ) 0) (Eq φ 0)
                                                                -/
theorem coe_eq_zero_iff : (φ : PowerSeries R) = 0 ↔ φ = 0 := by rw [← coe_zero, coe_inj]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
                                                               /-
                                                                 R : Type u_1
                                                                 inst✝ : CommSemiring R
                                                                 φ : Polynomial R
                                                                 ⊢ Iff (Eq (↑φ) 1) (Eq φ 1)
                                                               -/
theorem coe_eq_one_iff : (φ : PowerSeries R) = 1 ↔ φ = 1 := by rw [← coe_one, coe_inj]
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- The coercion from polynomials to power series
as a ring homomorphism.
-/
def coeToPowerSeries.ringHom : R[X] →+* PowerSeries R where
  toFun := (Coe.coe : R[X] → PowerSeries R)
  map_zero' := coe_zero
  map_one' := coe_one
  map_add' := coe_add
  map_mul' := coe_mul


@[simp]
theorem coeToPowerSeries.ringHom_apply : coeToPowerSeries.ringHom φ = φ :=
  rfl


@[simp, norm_cast]
theorem coe_pow (n : ℕ) : ((φ ^ n : R[X]) : PowerSeries R) = (φ : PowerSeries R) ^ n :=
  coeToPowerSeries.ringHom.map_pow _ _


theorem eval₂_C_X_eq_coe : φ.eval₂ (PowerSeries.C R) PowerSeries.X = ↑φ := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    φ : Polynomial R
    ⊢ Eq (Polynomial.eval₂ (PowerSeries.C R) PowerSeries.X φ) ↑φ
  -/
  nth_rw 2 [← eval₂_C_X (p := φ)]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    φ : Polynomial R
    ⊢ Eq (Polynomial.eval₂ (PowerSeries.C R) PowerSeries.X φ) ↑(Polynomial.eval₂ P …
  -/
  rw [← coeToPowerSeries.ringHom_apply, eval₂_eq_sum_range, eval₂_eq_sum_range, map_sum]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    φ : Polynomial R
    ⊢ Eq ((Finset.range (HAdd.hAdd φ.natDegree 1)).sum fun i => HMul.hMul ((PowerS …
  -/
  apply Finset.sum_congr rfl
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    φ : Polynomial R
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd φ.natDegree 1)) x → Eq  …
  -/
  intros
  rw [map_mul, map_pow, coeToPowerSeries.ringHom_apply,
    coeToPowerSeries.ringHom_apply, coe_C, coe_X]


/-- The coercion from polynomials to power series
as an algebra homomorphism.
-/
def coeToPowerSeries.algHom : R[X] →ₐ[R] PowerSeries A :=
  { (PowerSeries.map (algebraMap R A)).comp coeToPowerSeries.ringHom with
                             /-
                               R : Type u_1
                               inst✝² : CommSemiring R
                               φ ψ : Polynomial R
                               A : Type u_2
                               inst✝¹ : Semiring A
                               inst✝ : Algebra R A
                               r : R
                               ⊢ Eq ((↑↑__src✝).toFun ((algebraMap R (Polynomial R)) r)) ((algebraMap R (Powe …
                             -/
    commutes' := fun r => by simp [algebraMap_apply, PowerSeries.algebraMap_apply] }
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem coeToPowerSeries.algHom_apply :
    coeToPowerSeries.algHom A φ = PowerSeries.map (algebraMap R A) ↑φ :=
  rfl


@[simp, norm_cast]
lemma coe_neg (p : R[X]) : ((- p : R[X]) : PowerSeries R) = - p :=
  coeToPowerSeries.ringHom.map_neg p


@[simp, norm_cast]
lemma coe_sub (p q : R[X]) : ((p - q : R[X]) : PowerSeries R) = p - q :=
  coeToPowerSeries.ringHom.map_sub p q


instance algebraPolynomial : Algebra R[X] A⟦X⟧ :=
  RingHom.toAlgebra (Polynomial.coeToPowerSeries.algHom A).toRingHom


instance algebraPowerSeries : Algebra R⟦X⟧ A⟦X⟧ :=
  (map (algebraMap R A)).toAlgebra

-- see Note [lower instance priority]

instance (priority := 100) algebraPolynomial' {A : Type*} [CommSemiring A] [Algebra R A[X]] :
    Algebra R A⟦X⟧ :=
  RingHom.toAlgebra <| Polynomial.coeToPowerSeries.ringHom.comp (algebraMap R A[X])


theorem algebraMap_apply' (p : R[X]) : algebraMap R[X] A⟦X⟧ p = map (algebraMap R A) p :=
  rfl


theorem algebraMap_apply'' :
    algebraMap R⟦X⟧ A⟦X⟧ f = map (algebraMap R A) f :=
  rfl


