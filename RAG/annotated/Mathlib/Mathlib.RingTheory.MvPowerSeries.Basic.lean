/-- Multivariate formal power series, where `σ` is the index set of the variables
and `R` is the coefficient ring. -/
def MvPowerSeries (σ : Type*) (R : Type*) :=
  (σ →₀ ℕ) → R


instance [Inhabited R] : Inhabited (MvPowerSeries σ R) :=
  ⟨fun _ => default⟩


instance [Zero R] : Zero (MvPowerSeries σ R) :=
  Pi.instZero


instance [AddMonoid R] : AddMonoid (MvPowerSeries σ R) :=
  Pi.addMonoid


instance [AddGroup R] : AddGroup (MvPowerSeries σ R) :=
  Pi.addGroup


instance [AddCommMonoid R] : AddCommMonoid (MvPowerSeries σ R) :=
  Pi.addCommMonoid


instance [AddCommGroup R] : AddCommGroup (MvPowerSeries σ R) :=
  Pi.addCommGroup


instance [Nontrivial R] : Nontrivial (MvPowerSeries σ R) :=
  Function.nontrivial


instance {A} [Semiring R] [AddCommMonoid A] [Module R A] : Module R (MvPowerSeries σ A) :=
  Pi.module _ _ _


instance {A S} [Semiring R] [Semiring S] [AddCommMonoid A] [Module R A] [Module S A] [SMul R S]
    [IsScalarTower R S A] : IsScalarTower R S (MvPowerSeries σ A) :=
  Pi.isScalarTower


/-- The `n`th monomial as multivariate formal power series:
  it is defined as the `R`-linear map from `R` to the semi-ring
  of multivariate formal power series associating to each `a`
  the map sending `n : σ →₀ ℕ` to the value `a`
  and sending all other `x : σ →₀ ℕ` different from `n` to `0`. -/
def monomial (n : σ →₀ ℕ) : R →ₗ[R] MvPowerSeries σ R :=
  letI := Classical.decEq σ
  LinearMap.single R (fun _ ↦ R) n


/-- The `n`th coefficient of a multivariate formal power series. -/
def coeff (n : σ →₀ ℕ) : MvPowerSeries σ R →ₗ[R] R :=
  LinearMap.proj n


theorem coeff_apply (f : MvPowerSeries σ R) (d : σ →₀ ℕ) : coeff R d f = f d :=
  rfl


/-- Two multivariate formal power series are equal if all their coefficients are equal. -/
@[ext]
theorem ext {φ ψ} (h : ∀ n : σ →₀ ℕ, coeff R n φ = coeff R n ψ) : φ = ψ :=
  funext h


theorem monomial_def [DecidableEq σ] (n : σ →₀ ℕ) :
    (monomial R n) = LinearMap.single R (fun _ ↦ R) n := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : DecidableEq σ
    n : Finsupp σ Nat
    ⊢ Eq (MvPowerSeries.monomial R n) (LinearMap.single R (fun x => R) n)
  -/
  rw [monomial]
  -- unify the `Decidable` arguments
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : DecidableEq σ
    n : Finsupp σ Nat
    ⊢ Eq (LinearMap.single R (fun x => R) n) (LinearMap.single R (fun x => R) n)
  -/
  convert rfl
  /-
    🎉 no goals
  -/


theorem coeff_monomial [DecidableEq σ] (m n : σ →₀ ℕ) (a : R) :
    coeff R m (monomial R n a) = if m = n then a else 0 := by
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : DecidableEq σ
    m n : Finsupp σ Nat
    a : R
    ⊢ Eq ((MvPowerSeries.coeff R m) ((MvPowerSeries.monomial R n) a)) (ite (Eq m n …
  -/
  erw [coeff, monomial_def, LinearMap.proj_apply (i := m)]
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : DecidableEq σ
    m n : Finsupp σ Nat
    a : R
    ⊢ Eq ((LinearMap.single R (fun x => R) n) a m) (ite (Eq m n) a 0)
  -/
  erw [LinearMap.single_apply, Pi.single_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_monomial_same (n : σ →₀ ℕ) (a : R) : coeff R n (monomial R n a) = a := by
  classical
  rw [monomial_def]
  exact Pi.single_eq_same _ _


theorem coeff_monomial_ne {m n : σ →₀ ℕ} (h : m ≠ n) (a : R) : coeff R m (monomial R n a) = 0 := by
  classical
  rw [monomial_def]
  exact Pi.single_eq_of_ne h _


theorem eq_of_coeff_monomial_ne_zero {m n : σ →₀ ℕ} {a : R} (h : coeff R m (monomial R n a) ≠ 0) :
    m = n :=
  by_contra fun h' => h <| coeff_monomial_ne h' a


@[simp]
theorem coeff_comp_monomial (n : σ →₀ ℕ) : (coeff R n).comp (monomial R n) = LinearMap.id :=
  LinearMap.ext <| coeff_monomial_same n


@[simp]
theorem coeff_zero (n : σ →₀ ℕ) : coeff R n (0 : MvPowerSeries σ R) = 0 :=
  rfl


theorem eq_zero_iff_forall_coeff_zero {f : MvPowerSeries σ R} :
    f = 0 ↔ (∀ d : σ →₀ ℕ, coeff R d f = 0) :=
  MvPowerSeries.ext_iff


theorem ne_zero_iff_exists_coeff_ne_zero (f : MvPowerSeries σ R) :
    f ≠ 0 ↔ (∃ d : σ →₀ ℕ, coeff R d f ≠ 0) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    ⊢ Iff (Ne f 0) (Exists fun d => Ne ((MvPowerSeries.coeff R d) f) 0)
  -/
  simp only [MvPowerSeries.ext_iff, ne_eq, coeff_zero, not_forall]
  /-
    🎉 no goals
  -/


instance : One (MvPowerSeries σ R) :=
  ⟨monomial R (0 : σ →₀ ℕ) 1⟩


theorem coeff_one [DecidableEq σ] : coeff R n (1 : MvPowerSeries σ R) = if n = 0 then 1 else 0 :=
  coeff_monomial _ _ _


theorem coeff_zero_one : coeff R (0 : σ →₀ ℕ) 1 = 1 :=
  coeff_monomial_same 0 1


theorem monomial_zero_one : monomial R (0 : σ →₀ ℕ) 1 = 1 :=
  rfl


instance : AddMonoidWithOne (MvPowerSeries σ R) :=
                                          /-
                                            σ : Type u_1
                                            R : Type u_2
                                            inst✝ : Semiring R
                                            m n : Finsupp σ Nat
                                            φ ψ : MvPowerSeries σ R
                                            ⊢ AddMonoid (MvPowerSeries σ R)
                                          -/
  { show AddMonoid (MvPowerSeries σ R) by infer_instance with
                                          /-
                                            🎉 no goals
                                          -/
    natCast := fun n => monomial R 0 n
                       /-
                         σ : Type u_1
                         R : Type u_2
                         inst✝ : Semiring R
                         m n : Finsupp σ Nat
                         φ ψ : MvPowerSeries σ R
                         ⊢ Eq (NatCast.natCast 0) 0
                       -/
    natCast_zero := by simp [Nat.cast]
                       /-
                         🎉 no goals
                       -/
                       /-
                         σ : Type u_1
                         R : Type u_2
                         inst✝ : Semiring R
                         m n : Finsupp σ Nat
                         φ ψ : MvPowerSeries σ R
                         ⊢ ∀ (n : Nat), Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd (NatCast.natCas …
                       -/
    natCast_succ := by simp [Nat.cast, monomial_zero_one]
                       /-
                         🎉 no goals
                       -/
    one := 1 }


instance : Mul (MvPowerSeries σ R) :=
  letI := Classical.decEq σ
  ⟨fun φ ψ n => ∑ p ∈ antidiagonal n, coeff R p.1 φ * coeff R p.2 ψ⟩


theorem coeff_mul [DecidableEq σ] :
    coeff R n (φ * ψ) = ∑ p ∈ antidiagonal n, coeff R p.1 φ * coeff R p.2 ψ := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    n : Finsupp σ Nat
    φ ψ : MvPowerSeries σ R
    inst✝ : DecidableEq σ
    ⊢ Eq ((MvPowerSeries.coeff R n) (HMul.hMul φ ψ)) ((Finset.HasAntidiagonal.anti …
  -/
  refine Finset.sum_congr ?_ fun _ _ => rfl
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    n : Finsupp σ Nat
    φ ψ : MvPowerSeries σ R
    inst✝ : DecidableEq σ
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal n) (Finset.HasAntidiagonal.antidiago …
  -/
  rw [Subsingleton.elim (Classical.decEq σ) ‹DecidableEq σ›]
  /-
    🎉 no goals
  -/


protected theorem zero_mul : (0 : MvPowerSeries σ R) * φ = 0 :=
                  /-
                    σ : Type u_1
                    R : Type u_2
                    inst✝ : Semiring R
                    φ : MvPowerSeries σ R
                    n : Finsupp σ Nat
                    ⊢ Eq ((MvPowerSeries.coeff R n) (HMul.hMul 0 φ)) ((MvPowerSeries.coeff R n) 0)
                  -/
  ext fun n => by classical simp [coeff_mul]
                  /-
                    🎉 no goals
                  -/


protected theorem mul_zero : φ * 0 = 0 :=
                  /-
                    σ : Type u_1
                    R : Type u_2
                    inst✝ : Semiring R
                    φ : MvPowerSeries σ R
                    n : Finsupp σ Nat
                    ⊢ Eq ((MvPowerSeries.coeff R n) (HMul.hMul φ 0)) ((MvPowerSeries.coeff R n) 0)
                  -/
  ext fun n => by classical simp [coeff_mul]
                  /-
                    🎉 no goals
                  -/


theorem coeff_monomial_mul (a : R) :
    coeff R m (monomial R n a * φ) = if n ≤ m then a * coeff R (m - n) φ else 0 := by
  classical
  have :
    ∀ p ∈ antidiagonal m,
      coeff R (p : (σ →₀ ℕ) × (σ →₀ ℕ)).1 (monomial R n a) * coeff R p.2 φ ≠ 0 → p.1 = n :=
    fun p _ hp => eq_of_coeff_monomial_ne_zero (left_ne_zero_of_mul hp)
  rw [coeff_mul, ← Finset.sum_filter_of_ne this, Finset.filter_fst_eq_antidiagonal _ n,
    Finset.sum_ite_index]
  simp only [Finset.sum_singleton, coeff_monomial_same, Finset.sum_empty]


theorem coeff_mul_monomial (a : R) :
    coeff R m (φ * monomial R n a) = if n ≤ m then coeff R (m - n) φ * a else 0 := by
  classical
  have :
    ∀ p ∈ antidiagonal m,
      coeff R (p : (σ →₀ ℕ) × (σ →₀ ℕ)).1 φ * coeff R p.2 (monomial R n a) ≠ 0 → p.2 = n :=
    fun p _ hp => eq_of_coeff_monomial_ne_zero (right_ne_zero_of_mul hp)
  rw [coeff_mul, ← Finset.sum_filter_of_ne this, Finset.filter_snd_eq_antidiagonal _ n,
    Finset.sum_ite_index]
  simp only [Finset.sum_singleton, coeff_monomial_same, Finset.sum_empty]


theorem coeff_add_monomial_mul (a : R) :
    coeff R (m + n) (monomial R m a * φ) = a * coeff R n φ := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    m n : Finsupp σ Nat
    φ : MvPowerSeries σ R
    a : R
    ⊢ Eq ((MvPowerSeries.coeff R (HAdd.hAdd m n)) (HMul.hMul ((MvPowerSeries.monom …
  -/
  rw [coeff_monomial_mul, if_pos, add_tsub_cancel_left]
  /-
    case hc
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    m n : Finsupp σ Nat
    φ : MvPowerSeries σ R
    a : R
    ⊢ LE.le m (HAdd.hAdd m n)
  -/
  exact le_add_right le_rfl
  /-
    🎉 no goals
  -/


theorem coeff_add_mul_monomial (a : R) :
    coeff R (m + n) (φ * monomial R n a) = coeff R m φ * a := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    m n : Finsupp σ Nat
    φ : MvPowerSeries σ R
    a : R
    ⊢ Eq ((MvPowerSeries.coeff R (HAdd.hAdd m n)) (HMul.hMul φ ((MvPowerSeries.mon …
  -/
  rw [coeff_mul_monomial, if_pos, add_tsub_cancel_right]
  /-
    case hc
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    m n : Finsupp σ Nat
    φ : MvPowerSeries σ R
    a : R
    ⊢ LE.le n (HAdd.hAdd m n)
  -/
  exact le_add_left le_rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem commute_monomial {a : R} {n} :
    Commute φ (monomial R n a) ↔ ∀ m, Commute (coeff R m φ) a := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    φ : MvPowerSeries σ R
    a : R
    n : Finsupp σ Nat
    ⊢ Iff (Commute φ ((MvPowerSeries.monomial R n) a)) (∀ (m : Finsupp σ Nat), Com …
  -/
  rw [commute_iff_eq, MvPowerSeries.ext_iff]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    φ : MvPowerSeries σ R
    a : R
    n : Finsupp σ Nat
    ⊢ Iff (∀ (n_1 : Finsupp σ Nat), Eq ((MvPowerSeries.coeff R n_1) (HMul.hMul φ ( …
  -/
  refine ⟨fun h m => ?_, fun h m => ?_⟩
    /-
      case refine_1
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      φ : MvPowerSeries σ R
      a : R
      n : Finsupp σ Nat
      h : ∀ (n_1 : Finsupp σ Nat), Eq ((MvPowerSeries.coeff R n_1) (HMul.hMul φ ((Mv …
      m : Finsupp σ Nat
      ⊢ Commute ((MvPowerSeries.coeff R m) φ) a
    -/
  · have := h (m + n)
    /-
      case refine_1
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      φ : MvPowerSeries σ R
      a : R
      n : Finsupp σ Nat
      h : ∀ (n_1 : Finsupp σ Nat), Eq ((MvPowerSeries.coeff R n_1) (HMul.hMul φ ((Mv …
      m : Finsupp σ Nat
      this : Eq ((MvPowerSeries.coeff R (HAdd.hAdd m n)) (HMul.hMul φ ((MvPowerSerie …
      ⊢ Commute ((MvPowerSeries.coeff R m) φ) a
    -/
    rwa [coeff_add_mul_monomial, add_comm, coeff_add_monomial_mul] at this
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      φ : MvPowerSeries σ R
      a : R
      n : Finsupp σ Nat
      h : ∀ (m : Finsupp σ Nat), Commute ((MvPowerSeries.coeff R m) φ) a
      m : Finsupp σ Nat
      ⊢ Eq ((MvPowerSeries.coeff R m) (HMul.hMul φ ((MvPowerSeries.monomial R n) a)) …
    -/
  · rw [coeff_mul_monomial, coeff_monomial_mul]
    /-
      case refine_2
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      φ : MvPowerSeries σ R
      a : R
      n : Finsupp σ Nat
      h : ∀ (m : Finsupp σ Nat), Commute ((MvPowerSeries.coeff R m) φ) a
      m : Finsupp σ Nat
      ⊢ Eq (ite (LE.le n m) (HMul.hMul ((MvPowerSeries.coeff R (HSub.hSub m n)) φ) a …
    -/
    split_ifs <;> [apply h; rfl]
    /-
      🎉 no goals
    -/


protected theorem one_mul : (1 : MvPowerSeries σ R) * φ = φ :=
                  /-
                    σ : Type u_1
                    R : Type u_2
                    inst✝ : Semiring R
                    φ : MvPowerSeries σ R
                    n : Finsupp σ Nat
                    ⊢ Eq ((MvPowerSeries.coeff R n) (HMul.hMul 1 φ)) ((MvPowerSeries.coeff R n) φ)
                  -/
  ext fun n => by simpa using coeff_add_monomial_mul 0 n φ 1
                  /-
                    🎉 no goals
                  -/


protected theorem mul_one : φ * 1 = φ :=
                  /-
                    σ : Type u_1
                    R : Type u_2
                    inst✝ : Semiring R
                    φ : MvPowerSeries σ R
                    n : Finsupp σ Nat
                    ⊢ Eq ((MvPowerSeries.coeff R n) (HMul.hMul φ 1)) ((MvPowerSeries.coeff R n) φ)
                  -/
  ext fun n => by simpa using coeff_add_mul_monomial n 0 φ 1
                  /-
                    🎉 no goals
                  -/


protected theorem mul_add (φ₁ φ₂ φ₃ : MvPowerSeries σ R) : φ₁ * (φ₂ + φ₃) = φ₁ * φ₂ + φ₁ * φ₃ :=
  ext fun n => by
    /-
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      φ₁ φ₂ φ₃ : MvPowerSeries σ R
      n : Finsupp σ Nat
      ⊢ Eq ((MvPowerSeries.coeff R n) (HMul.hMul φ₁ (HAdd.hAdd φ₂ φ₃))) ((MvPowerSer …
    -/
    classical simp only [coeff_mul, mul_add, Finset.sum_add_distrib, LinearMap.map_add]
    /-
      🎉 no goals
    -/


protected theorem add_mul (φ₁ φ₂ φ₃ : MvPowerSeries σ R) : (φ₁ + φ₂) * φ₃ = φ₁ * φ₃ + φ₂ * φ₃ :=
  ext fun n => by
    /-
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      φ₁ φ₂ φ₃ : MvPowerSeries σ R
      n : Finsupp σ Nat
      ⊢ Eq ((MvPowerSeries.coeff R n) (HMul.hMul (HAdd.hAdd φ₁ φ₂) φ₃)) ((MvPowerSer …
    -/
    classical simp only [coeff_mul, add_mul, Finset.sum_add_distrib, LinearMap.map_add]
    /-
      🎉 no goals
    -/


protected theorem mul_assoc (φ₁ φ₂ φ₃ : MvPowerSeries σ R) : φ₁ * φ₂ * φ₃ = φ₁ * (φ₂ * φ₃) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    φ₁ φ₂ φ₃ : MvPowerSeries σ R
    ⊢ Eq (HMul.hMul (HMul.hMul φ₁ φ₂) φ₃) (HMul.hMul φ₁ (HMul.hMul φ₂ φ₃))
  -/
  ext1 n
  classical
  simp only [coeff_mul, Finset.sum_mul, Finset.mul_sum, Finset.sum_sigma']
  apply Finset.sum_nbij' (fun ⟨⟨_i, j⟩, ⟨k, l⟩⟩ ↦ ⟨(k, l + j), (l, j)⟩)
    (fun ⟨⟨i, _j⟩, ⟨k, l⟩⟩ ↦ ⟨(i + k, l), (i, k)⟩) <;> aesop (add simp [add_assoc, mul_assoc])


instance : Semiring (MvPowerSeries σ R) :=
  { inferInstanceAs (AddMonoidWithOne (MvPowerSeries σ R)),
    inferInstanceAs (Mul (MvPowerSeries σ R)),
    inferInstanceAs (AddCommMonoid (MvPowerSeries σ R)) with
    mul_one := MvPowerSeries.mul_one
    one_mul := MvPowerSeries.one_mul
    mul_assoc := MvPowerSeries.mul_assoc
    mul_zero := MvPowerSeries.mul_zero
    zero_mul := MvPowerSeries.zero_mul
    left_distrib := MvPowerSeries.mul_add
    right_distrib := MvPowerSeries.add_mul }


instance [CommSemiring R] : CommSemiring (MvPowerSeries σ R) :=
                                         /-
                                           σ : Type u_1
                                           R : Type u_2
                                           inst✝ : CommSemiring R
                                           ⊢ Semiring (MvPowerSeries σ R)
                                         -/
  { show Semiring (MvPowerSeries σ R) by infer_instance with
                                         /-
                                           🎉 no goals
                                         -/
    mul_comm := fun φ ψ =>
      ext fun n => by
        classical
        simpa only [coeff_mul, mul_comm] using
          sum_antidiagonal_swap n fun a b => coeff R a φ * coeff R b ψ }


instance [Ring R] : Ring (MvPowerSeries σ R) :=
  { inferInstanceAs (Semiring (MvPowerSeries σ R)),
    inferInstanceAs (AddCommGroup (MvPowerSeries σ R)) with }


instance [CommRing R] : CommRing (MvPowerSeries σ R) :=
  { inferInstanceAs (CommSemiring (MvPowerSeries σ R)),
    inferInstanceAs (AddCommGroup (MvPowerSeries σ R)) with }


theorem monomial_mul_monomial (m n : σ →₀ ℕ) (a b : R) :
    monomial R m a * monomial R n b = monomial R (m + n) (a * b) := by
  classical
  ext k
  simp only [coeff_mul_monomial, coeff_monomial]
  split_ifs with h₁ h₂ h₃ h₃ h₂ <;> try rfl
  · rw [← h₂, tsub_add_cancel_of_le h₁] at h₃
    exact (h₃ rfl).elim
  · rw [h₃, add_tsub_cancel_right] at h₂
    exact (h₂ rfl).elim
  · exact zero_mul b
  · rw [h₂] at h₁
    exact (h₁ <| le_add_left le_rfl).elim


/-- The constant multivariate formal power series. -/
def C : R →+* MvPowerSeries σ R :=
  { monomial R (0 : σ →₀ ℕ) with
    map_one' := rfl
    map_mul' := fun a b => (monomial_mul_monomial 0 0 a b).symm
    map_zero' := (monomial R (0 : _)).map_zero }


@[simp]
theorem monomial_zero_eq_C : ⇑(monomial R (0 : σ →₀ ℕ)) = C σ R :=
  rfl


theorem monomial_zero_eq_C_apply (a : R) : monomial R (0 : σ →₀ ℕ) a = C σ R a :=
  rfl


theorem coeff_C [DecidableEq σ] (n : σ →₀ ℕ) (a : R) :
    coeff R n (C σ R a) = if n = 0 then a else 0 :=
  coeff_monomial _ _ _


theorem coeff_zero_C (a : R) : coeff R (0 : σ →₀ ℕ) (C σ R a) = a :=
  coeff_monomial_same 0 a


/-- The variables of the multivariate formal power series ring. -/
def X (s : σ) : MvPowerSeries σ R :=
  monomial R (single s 1) 1


theorem coeff_X [DecidableEq σ] (n : σ →₀ ℕ) (s : σ) :
    coeff R n (X s : MvPowerSeries σ R) = if n = single s 1 then 1 else 0 :=
  coeff_monomial _ _ _


theorem coeff_index_single_X [DecidableEq σ] (s t : σ) :
    coeff R (single t 1) (X s : MvPowerSeries σ R) = if t = s then 1 else 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : DecidableEq σ
    s t : σ
    ⊢ Eq ((MvPowerSeries.coeff R (Finsupp.single t 1)) (MvPowerSeries.X s)) (ite ( …
  -/
  simp only [coeff_X, single_left_inj (one_ne_zero : (1 : ℕ) ≠ 0)]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_index_single_self_X (s : σ) : coeff R (single s 1) (X s : MvPowerSeries σ R) = 1 :=
  coeff_monomial_same _ _


theorem coeff_zero_X (s : σ) : coeff R (0 : σ →₀ ℕ) (X s : MvPowerSeries σ R) = 0 := by
  classical
  rw [coeff_X, if_neg]
  intro h
  exact one_ne_zero (single_eq_zero.mp h.symm)


theorem commute_X (φ : MvPowerSeries σ R) (s : σ) : Commute φ (X s) :=
  φ.commute_monomial.mpr fun _m => Commute.one_right _


theorem X_def (s : σ) : X s = monomial R (single s 1) 1 :=
  rfl


theorem X_pow_eq (s : σ) (n : ℕ) : (X s : MvPowerSeries σ R) ^ n = monomial R (single s n) 1 := by
  induction n with
  | zero => simp
  | succ n ih => rw [pow_succ, ih, Finsupp.single_add, X, monomial_mul_monomial, one_mul]


theorem coeff_X_pow [DecidableEq σ] (m : σ →₀ ℕ) (s : σ) (n : ℕ) :
    coeff R m ((X s : MvPowerSeries σ R) ^ n) = if m = single s n then 1 else 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : DecidableEq σ
    m : Finsupp σ Nat
    s : σ
    n : Nat
    ⊢ Eq ((MvPowerSeries.coeff R m) (HPow.hPow (MvPowerSeries.X s) n)) (ite (Eq m  …
  -/
  rw [X_pow_eq s n, coeff_monomial]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_mul_C (n : σ →₀ ℕ) (φ : MvPowerSeries σ R) (a : R) :
                                                    /-
                                                      σ : Type u_1
                                                      R : Type u_2
                                                      inst✝ : Semiring R
                                                      n : Finsupp σ Nat
                                                      φ : MvPowerSeries σ R
                                                      a : R
                                                      ⊢ Eq ((MvPowerSeries.coeff R n) (HMul.hMul φ ((MvPowerSeries.C σ R) a))) (HMul …
                                                    -/
    coeff R n (φ * C σ R a) = coeff R n φ * a := by simpa using coeff_add_mul_monomial n 0 φ a
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem coeff_C_mul (n : σ →₀ ℕ) (φ : MvPowerSeries σ R) (a : R) :
                                                    /-
                                                      σ : Type u_1
                                                      R : Type u_2
                                                      inst✝ : Semiring R
                                                      n : Finsupp σ Nat
                                                      φ : MvPowerSeries σ R
                                                      a : R
                                                      ⊢ Eq ((MvPowerSeries.coeff R n) (HMul.hMul ((MvPowerSeries.C σ R) a) φ)) (HMul …
                                                    -/
    coeff R n (C σ R a * φ) = a * coeff R n φ := by simpa using coeff_add_monomial_mul 0 n φ a
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem coeff_zero_mul_X (φ : MvPowerSeries σ R) (s : σ) : coeff R (0 : σ →₀ ℕ) (φ * X s) = 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    φ : MvPowerSeries σ R
    s : σ
    ⊢ Eq ((MvPowerSeries.coeff R 0) (HMul.hMul φ (MvPowerSeries.X s))) 0
  -/
  have : ¬single s 1 ≤ 0 := fun h => by simpa using h s
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    φ : MvPowerSeries σ R
    s : σ
    this : Not (LE.le (Finsupp.single s 1) 0)
    ⊢ Eq ((MvPowerSeries.coeff R 0) (HMul.hMul φ (MvPowerSeries.X s))) 0
  -/
  simp only [X, coeff_mul_monomial, if_neg this]
  /-
    🎉 no goals
  -/


theorem coeff_zero_X_mul (φ : MvPowerSeries σ R) (s : σ) : coeff R (0 : σ →₀ ℕ) (X s * φ) = 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    φ : MvPowerSeries σ R
    s : σ
    ⊢ Eq ((MvPowerSeries.coeff R 0) (HMul.hMul (MvPowerSeries.X s) φ)) 0
  -/
  rw [← (φ.commute_X s).eq, coeff_zero_mul_X]
  /-
    🎉 no goals
  -/


/-- The constant coefficient of a formal power series. -/
def constantCoeff : MvPowerSeries σ R →+* R :=
  { coeff R (0 : σ →₀ ℕ) with
    toFun := coeff R (0 : σ →₀ ℕ)
    map_one' := coeff_zero_one
                              /-
                                σ : Type u_1
                                R : Type u_2
                                inst✝ : Semiring R
                                φ ψ : MvPowerSeries σ R
                                ⊢ Eq ({ toFun := ⇑(MvPowerSeries.coeff R 0), map_one' := ⋯ }.toFun (HMul.hMul  …
                              -/
    map_mul' := fun φ ψ => by classical simp [coeff_mul, support_single_ne_zero]
                              /-
                                🎉 no goals
                              -/
    map_zero' := LinearMap.map_zero _ }


@[simp]
theorem coeff_zero_eq_constantCoeff : ⇑(coeff R (0 : σ →₀ ℕ)) = constantCoeff σ R :=
  rfl


theorem coeff_zero_eq_constantCoeff_apply (φ : MvPowerSeries σ R) :
    coeff R (0 : σ →₀ ℕ) φ = constantCoeff σ R φ :=
  rfl


@[simp]
theorem constantCoeff_C (a : R) : constantCoeff σ R (C σ R a) = a :=
  rfl


@[simp]
theorem constantCoeff_comp_C : (constantCoeff σ R).comp (C σ R) = RingHom.id R :=
  rfl


@[simp]
theorem constantCoeff_zero : constantCoeff σ R 0 = 0 :=
  rfl


@[simp]
theorem constantCoeff_one : constantCoeff σ R 1 = 1 :=
  rfl


@[simp]
theorem constantCoeff_X (s : σ) : constantCoeff σ R (X s) = 0 :=
  coeff_zero_X s


/-- If a multivariate formal power series is invertible,
 then so is its constant coefficient. -/
theorem isUnit_constantCoeff (φ : MvPowerSeries σ R) (h : IsUnit φ) :
    IsUnit (constantCoeff σ R φ) :=
  h.map _


@[simp]
theorem coeff_smul (f : MvPowerSeries σ R) (n) (a : R) : coeff _ n (a • f) = a * coeff _ n f :=
  rfl


theorem smul_eq_C_mul (f : MvPowerSeries σ R) (a : R) : a • f = C σ R a * f := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    a : R
    ⊢ Eq (HSMul.hSMul a f) (HMul.hMul ((MvPowerSeries.C σ R) a) f)
  -/
  ext
  /-
    case h
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    a : R
    n✝ : Finsupp σ Nat
    ⊢ Eq ((MvPowerSeries.coeff R n✝) (HSMul.hSMul a f)) ((MvPowerSeries.coeff R n✝ …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem X_inj [Nontrivial R] {s t : σ} : (X s : MvPowerSeries σ R) = X t ↔ s = t :=
  ⟨by
    classical
    intro h
    replace h := congr_arg (coeff R (single s 1)) h
    rw [coeff_X, if_pos rfl, coeff_X] at h
    split_ifs at h with H
    · rw [Finsupp.single_eq_single_iff] at H
      cases' H with H H
      · exact H.1
      · exfalso
        exact one_ne_zero H.1
    · exfalso
      exact one_ne_zero h, congr_arg X⟩


/-- The map between multivariate formal power series induced by a map on the coefficients. -/
def map : MvPowerSeries σ R →+* MvPowerSeries σ S where
  toFun φ n := f <| coeff R n φ
  map_zero' := ext fun _n => f.map_zero
  map_one' :=
    ext fun n =>
      show f ((coeff R n) 1) = (coeff S n) 1 by
        classical
        rw [coeff_one, coeff_one]
        split_ifs with h
        · simp only [RingHom.map_ite_one_zero, ite_true, map_one, h]
        · simp only [RingHom.map_ite_one_zero, ite_false, map_zero, h]
  map_add' φ ψ :=
                                                                                         /-
                                                                                           σ : Type u_1
                                                                                           R : Type u_2
                                                                                           S : Type u_3
                                                                                           T : Type u_4
                                                                                           inst✝² : Semiring R
                                                                                           inst✝¹ : Semiring S
                                                                                           inst✝ : Semiring T
                                                                                           f : RingHom R S
                                                                                           g : RingHom S T
                                                                                           φ ψ : MvPowerSeries σ R
                                                                                           n : Finsupp σ Nat
                                                                                           ⊢ Eq (f ((MvPowerSeries.coeff R n) (HAdd.hAdd φ ψ))) (HAdd.hAdd (f ((MvPowerSe …
                                                                                         -/
    ext fun n => show f ((coeff R n) (φ + ψ)) = f ((coeff R n) φ) + f ((coeff R n) ψ) by simp
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
  map_mul' φ ψ :=
    ext fun n =>
      show f _ = _ by
        classical
        rw [coeff_mul, map_sum, coeff_mul]
        apply Finset.sum_congr rfl
        rintro ⟨i, j⟩ _; rw [f.map_mul]; rfl


@[simp]
theorem map_id : map σ (RingHom.id R) = RingHom.id _ :=
  rfl


theorem map_comp : map σ (g.comp f) = (map σ g).comp (map σ f) :=
  rfl


@[simp]
theorem coeff_map (n : σ →₀ ℕ) (φ : MvPowerSeries σ R) : coeff S n (map σ f φ) = f (coeff R n φ) :=
  rfl


@[simp]
theorem constantCoeff_map (φ : MvPowerSeries σ R) :
    constantCoeff σ S (map σ f φ) = f (constantCoeff σ R φ) :=
  rfl


@[simp]
theorem map_monomial (n : σ →₀ ℕ) (a : R) : map σ f (monomial R n a) = monomial S n (f a) := by
  classical
  ext m
  simp [coeff_monomial, apply_ite f]


@[simp]
theorem map_C (a : R) : map σ f (C σ R a) = C σ S (f a) :=
  map_monomial _ _ _


@[simp]
                                                  /-
                                                    σ : Type u_1
                                                    R : Type u_2
                                                    S : Type u_3
                                                    inst✝¹ : Semiring R
                                                    inst✝ : Semiring S
                                                    f : RingHom R S
                                                    s : σ
                                                    ⊢ Eq ((MvPowerSeries.map σ f) (MvPowerSeries.X s)) (MvPowerSeries.X s)
                                                  -/
theorem map_X (s : σ) : map σ f (X s) = X s := by simp [MvPowerSeries.X]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem X_pow_dvd_iff {s : σ} {n : ℕ} {φ : MvPowerSeries σ R} :
    (X s : MvPowerSeries σ R) ^ n ∣ φ ↔ ∀ m : σ →₀ ℕ, m s < n → coeff R m φ = 0 := by
  classical
  constructor
  · rintro ⟨φ, rfl⟩ m h
    rw [coeff_mul, Finset.sum_eq_zero]
    rintro ⟨i, j⟩ hij
    rw [coeff_X_pow, if_neg, zero_mul]
    contrapose! h
    dsimp at h
    subst i
    rw [mem_antidiagonal] at hij
    rw [← hij, Finsupp.add_apply, Finsupp.single_eq_same]
    exact Nat.le_add_right n _
  · intro h
    refine ⟨fun m => coeff R (m + single s n) φ, ?_⟩
    ext m
    by_cases H : m - single s n + single s n = m
    · rw [coeff_mul, Finset.sum_eq_single (single s n, m - single s n)]
      · rw [coeff_X_pow, if_pos rfl, one_mul]
        simpa using congr_arg (fun m : σ →₀ ℕ => coeff R m φ) H.symm
      · rintro ⟨i, j⟩ hij hne
        rw [mem_antidiagonal] at hij
        rw [coeff_X_pow]
        split_ifs with hi
        · exfalso
          apply hne
          rw [← hij, ← hi, Prod.mk.inj_iff]
          refine ⟨rfl, ?_⟩
          ext t
          simp only [add_tsub_cancel_left, Finsupp.add_apply, Finsupp.tsub_apply]
        · exact zero_mul _
      · intro hni
        exfalso
        apply hni
        rwa [mem_antidiagonal, add_comm]
    · rw [h, coeff_mul, Finset.sum_eq_zero]
      · rintro ⟨i, j⟩ hij
        rw [mem_antidiagonal] at hij
        rw [coeff_X_pow]
        split_ifs with hi
        · exfalso
          apply H
          rw [← hij, hi]
          ext
          rw [coe_add, coe_add, Pi.add_apply, Pi.add_apply, add_tsub_cancel_left, add_comm]
        · exact zero_mul _
      · contrapose! H
        ext t
        by_cases hst : s = t
        · subst t
          simpa using tsub_add_cancel_of_le H
        · simp [Finsupp.single_apply, hst]


theorem X_dvd_iff {s : σ} {φ : MvPowerSeries σ R} :
    (X s : MvPowerSeries σ R) ∣ φ ↔ ∀ m : σ →₀ ℕ, m s = 0 → coeff R m φ = 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    s : σ
    φ : MvPowerSeries σ R
    ⊢ Iff (Dvd.dvd (MvPowerSeries.X s) φ) (∀ (m : Finsupp σ Nat), Eq (m s) 0 → Eq  …
  -/
  rw [← pow_one (X s : MvPowerSeries σ R), X_pow_dvd_iff]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    s : σ
    φ : MvPowerSeries σ R
    ⊢ Iff (∀ (m : Finsupp σ Nat), LT.lt (m s) 1 → Eq ((MvPowerSeries.coeff R m) φ) …
  -/
  constructor <;> intro h m hm
    /-
      case mp
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      s : σ
      φ : MvPowerSeries σ R
      h : ∀ (m : Finsupp σ Nat), LT.lt (m s) 1 → Eq ((MvPowerSeries.coeff R m) φ) 0
      m : Finsupp σ Nat
      hm : Eq (m s) 0
      ⊢ Eq ((MvPowerSeries.coeff R m) φ) 0
    -/
  · exact h m (hm.symm ▸ zero_lt_one)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      s : σ
      φ : MvPowerSeries σ R
      h : ∀ (m : Finsupp σ Nat), Eq (m s) 0 → Eq ((MvPowerSeries.coeff R m) φ) 0
      m : Finsupp σ Nat
      hm : LT.lt (m s) 1
      ⊢ Eq ((MvPowerSeries.coeff R m) φ) 0
    -/
  · exact h m (Nat.eq_zero_of_le_zero <| Nat.le_of_succ_le_succ hm)
    /-
      🎉 no goals
    -/


/-- Coefficients of a product of power series -/
theorem coeff_prod [DecidableEq σ]
    (f : ι → MvPowerSeries σ R) (d : σ →₀ ℕ) (s : Finset ι) :
    coeff R d (∏ j ∈ s, f j) =
      ∑ l ∈ finsuppAntidiag s d,
        ∏ i ∈ s, coeff R (l i) (f i) := by
  induction s using Finset.induction_on generalizing d with
  | empty =>
    simp only [prod_empty, sum_const, nsmul_eq_mul, mul_one, coeff_one, finsuppAntidiag_empty]
    split_ifs
    · simp only [card_singleton, Nat.cast_one]
    · simp only [card_empty, Nat.cast_zero]
  | @insert a s ha ih =>
    rw [finsuppAntidiag_insert ha, prod_insert ha, coeff_mul, sum_biUnion]
    · apply Finset.sum_congr rfl
      simp only [mem_antidiagonal, sum_map, Function.Embedding.coeFn_mk, coe_update, Prod.forall]
      rintro u v rfl
      rw [ih, Finset.mul_sum, ← Finset.sum_attach]
      apply Finset.sum_congr rfl
      simp only [mem_attach, Finset.prod_insert ha, Function.update_self, forall_true_left,
        Subtype.forall]
      rintro x -
      rw [Finset.prod_congr rfl]
      intro i hi
      rw [Function.update_of_ne]
      exact ne_of_mem_of_not_mem hi ha
    · simp only [Set.PairwiseDisjoint, Set.Pairwise, mem_coe, mem_antidiagonal, ne_eq,
        disjoint_left, mem_map, mem_attach, Function.Embedding.coeFn_mk, true_and, Subtype.exists,
        exists_prop, not_exists, not_and, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂,
        Prod.forall, Prod.mk.injEq]
      rintro u v rfl u' v' huv h k - l - hkl
      obtain rfl : u' = u := by
        simpa only [Finsupp.coe_update, Function.update_self] using DFunLike.congr_fun hkl a
      simp only [add_right_inj] at huv
      exact h rfl huv.symm


/-- The `d`th coefficient of a power of a multivariate power series
is the sum, indexed by `finsuppAntidiag (Finset.range n) d`, of products of coefficients  -/
theorem coeff_pow [DecidableEq σ] (f : MvPowerSeries σ R) {n : ℕ} (d : σ →₀ ℕ) :
    coeff R d (f ^ n) =
      ∑ l in finsuppAntidiag (Finset.range n) d,
        ∏ i in Finset.range n, coeff R (l i) f := by
  suffices f ^ n = (Finset.range n).prod fun _ ↦ f by
    rw [this, coeff_prod]
  /-
    σ : Type u_1
    R : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    f : MvPowerSeries σ R
    n : Nat
    d : Finsupp σ Nat
    ⊢ Eq (HPow.hPow f n) ((Finset.range n).prod fun x => f)
  -/
  rw [Finset.prod_const, card_range]
  /-
    🎉 no goals
  -/


/-- Vanishing of coefficients of powers of multivariate power series
when the constant coefficient is nilpotent
[N. Bourbaki, *Algebra {II}*, Chapter 4, §4, n°2, proposition 3][bourbaki1981] -/
theorem coeff_eq_zero_of_constantCoeff_nilpotent
    {f : MvPowerSeries σ R} {m : ℕ} (hf : constantCoeff σ R f ^ m = 0)
    {d : σ →₀ ℕ} {n : ℕ} (hn : m + degree d ≤ n) : coeff R d (f ^ n) = 0 := by
  classical
  rw [coeff_pow]
  apply sum_eq_zero
  intro k hk
  rw [mem_finsuppAntidiag] at hk
  set s := {i ∈ range n | k i = 0} with hs_def
  have hs : s ⊆ range n := filter_subset _ _
  have hs' (i : ℕ) (hi : i ∈ s) : coeff R (k i) f = constantCoeff σ R f := by
    simp only [hs_def, mem_filter] at hi
    rw [hi.2, coeff_zero_eq_constantCoeff]
  have hs'' (i : ℕ) (hi : i ∈ s) : k i = 0 := by
    simp only [hs_def, mem_filter] at hi
    rw [hi.2]
  rw [← prod_sdiff (s₁ := s) (filter_subset _ _)]
  apply mul_eq_zero_of_right
  rw [prod_congr rfl hs', prod_const]
  suffices m ≤ #s by
    obtain ⟨m', hm'⟩ := Nat.exists_eq_add_of_le this
    rw [hm', pow_add, hf, MulZeroClass.zero_mul]
  rw [← Nat.add_le_add_iff_right, add_comm #s,
    Finset.card_sdiff_add_card_eq_card (filter_subset _ _), card_range]
  apply le_trans _ hn
  simp only [add_comm m, Nat.add_le_add_iff_right, ← hk.1,
    ← sum_sdiff (hs), sum_eq_zero (s := s) hs'', add_zero]
  rw [← hs_def]
  convert Finset.card_nsmul_le_sum (range n \ s) (fun x ↦ degree (k x)) 1 _
  · simp only [Algebra.id.smul_eq_mul, mul_one]
  · simp only [degree_eq_weight_one, map_sum]
  · simp only [hs_def, mem_filter, mem_sdiff, mem_range, not_and, and_imp]
    intro i hi hi'
    rw [← not_lt, Nat.lt_one_iff, degree_eq_zero_iff]
    exact hi' hi


instance : Algebra R (MvPowerSeries σ A) :=
  {
                                         /-
                                           σ : Type u_1
                                           R : Type u_2
                                           A : Type u_3
                                           inst✝² : CommSemiring R
                                           inst✝¹ : Semiring A
                                           inst✝ : Algebra R A
                                           ⊢ Module R (MvPowerSeries σ A)
                                         -/
    show Module R (MvPowerSeries σ A) by infer_instance with
                                         /-
                                           🎉 no goals
                                         -/
    commutes' := fun a φ => by
      /-
        σ : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        a : R
        φ : MvPowerSeries σ A
        ⊢ Eq (HMul.hMul (((MvPowerSeries.map σ (algebraMap R A)).comp (MvPowerSeries.C …
      -/
      ext n
      /-
        case h
        σ : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        a : R
        φ : MvPowerSeries σ A
        n : Finsupp σ Nat
        ⊢ Eq ((MvPowerSeries.coeff A n) (HMul.hMul (((MvPowerSeries.map σ (algebraMap  …
      -/
      simp [Algebra.commutes]
      /-
        🎉 no goals
      -/
    smul_def' := fun a σ => by
      /-
        σ✝ : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        a : R
        σ : MvPowerSeries σ✝ A
        ⊢ Eq (HSMul.hSMul a σ) (HMul.hMul (((MvPowerSeries.map σ✝ (algebraMap R A)).co …
      -/
      ext n
      /-
        case h
        σ✝ : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        a : R
        σ : MvPowerSeries σ✝ A
        n : Finsupp σ✝ Nat
        ⊢ Eq ((MvPowerSeries.coeff A n) (HSMul.hSMul a σ)) ((MvPowerSeries.coeff A n)  …
      -/
      simp [(coeff A n).map_smul_of_tower a, Algebra.smul_def]
      /-
        🎉 no goals
      -/
    toRingHom := (MvPowerSeries.map σ (algebraMap R A)).comp (C σ R) }


theorem c_eq_algebraMap : C σ R = algebraMap R (MvPowerSeries σ R) :=
  rfl


theorem algebraMap_apply {r : R} :
    algebraMap R (MvPowerSeries σ A) r = C σ A (algebraMap R A r) := by
  /-
    σ : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    r : R
    ⊢ Eq ((algebraMap R (MvPowerSeries σ A)) r) ((MvPowerSeries.C σ A) ((algebraMa …
  -/
  change (MvPowerSeries.map σ (algebraMap R A)).comp (C σ R) r = _
  /-
    σ : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    r : R
    ⊢ Eq (((MvPowerSeries.map σ (algebraMap R A)).comp (MvPowerSeries.C σ R)) r) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


instance [Nonempty σ] [Nontrivial R] : Nontrivial (Subalgebra R (MvPowerSeries σ R)) :=
  ⟨⟨⊥, ⊤, by
      classical
      rw [Ne, SetLike.ext_iff, not_forall]
      inhabit σ
      refine ⟨X default, ?_⟩
      simp only [Algebra.mem_bot, not_exists, Set.mem_range, iff_true, Algebra.mem_top]
      intro x
      rw [MvPowerSeries.ext_iff, not_forall]
      refine ⟨Finsupp.single default 1, ?_⟩
      simp [algebraMap_apply, coeff_C]⟩⟩


/-- The natural inclusion from multivariate polynomials into multivariate formal power series. -/
@[coe]
def toMvPowerSeries : MvPolynomial σ R → MvPowerSeries σ R :=
  fun φ n => coeff n φ


/-- The natural inclusion from multivariate polynomials into multivariate formal power series. -/
instance coeToMvPowerSeries : Coe (MvPolynomial σ R) (MvPowerSeries σ R) :=
  ⟨toMvPowerSeries⟩


theorem coe_def : (φ : MvPowerSeries σ R) = fun n => coeff n φ :=
  rfl


@[simp, norm_cast]
theorem coeff_coe (n : σ →₀ ℕ) : MvPowerSeries.coeff R n ↑φ = coeff n φ :=
  rfl


@[simp, norm_cast]
theorem coe_monomial (n : σ →₀ ℕ) (a : R) :
    (monomial n a : MvPowerSeries σ R) = MvPowerSeries.monomial R n a :=
  MvPowerSeries.ext fun m => by
    classical
    rw [coeff_coe, coeff_monomial, MvPowerSeries.coeff_monomial]
    split_ifs with h₁ h₂ <;> first |rfl|subst m; contradiction


@[simp, norm_cast]
theorem coe_zero : ((0 : MvPolynomial σ R) : MvPowerSeries σ R) = 0 :=
  rfl


@[simp, norm_cast]
theorem coe_one : ((1 : MvPolynomial σ R) : MvPowerSeries σ R) = 1 :=
    coe_monomial _ _


@[simp, norm_cast]
theorem coe_add : ((φ + ψ : MvPolynomial σ R) : MvPowerSeries σ R) = φ + ψ :=
  rfl


@[simp, norm_cast]
theorem coe_mul : ((φ * ψ : MvPolynomial σ R) : MvPowerSeries σ R) = φ * ψ :=
  MvPowerSeries.ext fun n => by
    classical
    simp only [coeff_coe, MvPowerSeries.coeff_mul, coeff_mul]


@[simp, norm_cast]
theorem coe_C (a : R) : ((C a : MvPolynomial σ R) : MvPowerSeries σ R) = MvPowerSeries.C σ R a :=
  coe_monomial _ _


@[simp, norm_cast]
theorem coe_X (s : σ) : ((X s : MvPolynomial σ R) : MvPowerSeries σ R) = MvPowerSeries.X s :=
  coe_monomial _ _


theorem coe_injective : Function.Injective (Coe.coe : MvPolynomial σ R → MvPowerSeries σ R) :=
    fun x y h => by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : CommSemiring R
    x y : MvPolynomial σ R
    h : Eq (Coe.coe x) (Coe.coe y)
    ⊢ Eq x y
  -/
  ext
  /-
    case a
    σ : Type u_1
    R : Type u_2
    inst✝ : CommSemiring R
    x y : MvPolynomial σ R
    h : Eq (Coe.coe x) (Coe.coe y)
    m✝ : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff m✝ x) (MvPolynomial.coeff m✝ y)
  -/
  simp_rw [← coeff_coe]
  /-
    case a
    σ : Type u_1
    R : Type u_2
    inst✝ : CommSemiring R
    x y : MvPolynomial σ R
    h : Eq (Coe.coe x) (Coe.coe y)
    m✝ : Finsupp σ Nat
    ⊢ Eq ((MvPowerSeries.coeff R m✝) ↑x) ((MvPowerSeries.coeff R m✝) ↑y)
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_inj : (φ : MvPowerSeries σ R) = ψ ↔ φ = ψ :=
  (coe_injective σ R).eq_iff


@[simp]
                                                                    /-
                                                                      σ : Type u_1
                                                                      R : Type u_2
                                                                      inst✝ : CommSemiring R
                                                                      φ : MvPolynomial σ R
                                                                      ⊢ Iff (Eq (↑φ) 0) (Eq φ 0)
                                                                    -/
theorem coe_eq_zero_iff : (φ : MvPowerSeries σ R) = 0 ↔ φ = 0 := by rw [← coe_zero, coe_inj]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
                                                                   /-
                                                                     σ : Type u_1
                                                                     R : Type u_2
                                                                     inst✝ : CommSemiring R
                                                                     φ : MvPolynomial σ R
                                                                     ⊢ Iff (Eq (↑φ) 1) (Eq φ 1)
                                                                   -/
theorem coe_eq_one_iff : (φ : MvPowerSeries σ R) = 1 ↔ φ = 1 := by rw [← coe_one, coe_inj]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- The coercion from multivariate polynomials to multivariate power series
as a ring homomorphism.
-/
def coeToMvPowerSeries.ringHom : MvPolynomial σ R →+* MvPowerSeries σ R where
  toFun := (Coe.coe : MvPolynomial σ R → MvPowerSeries σ R)
  map_zero' := coe_zero
  map_one' := coe_one
  map_add' := coe_add
  map_mul' := coe_mul


@[simp, norm_cast]
theorem coe_pow (n : ℕ) :
    ((φ ^ n : MvPolynomial σ R) : MvPowerSeries σ R) = (φ : MvPowerSeries σ R) ^ n :=
  coeToMvPowerSeries.ringHom.map_pow _ _


@[simp]
theorem coeToMvPowerSeries.ringHom_apply : coeToMvPowerSeries.ringHom φ = φ :=
  rfl


/-- The coercion from multivariate polynomials to multivariate power series
as an algebra homomorphism.
-/
def coeToMvPowerSeries.algHom : MvPolynomial σ R →ₐ[R] MvPowerSeries σ A :=
  { (MvPowerSeries.map σ (algebraMap R A)).comp coeToMvPowerSeries.ringHom with
                             /-
                               σ : Type u_1
                               R : Type u_2
                               inst✝² : CommSemiring R
                               φ ψ : MvPolynomial σ R
                               A : Type u_3
                               inst✝¹ : CommSemiring A
                               inst✝ : Algebra R A
                               r : R
                               ⊢ Eq ((↑↑__src✝).toFun ((algebraMap R (MvPolynomial σ R)) r)) ((algebraMap R ( …
                             -/
    commutes' := fun r => by simp [algebraMap_apply, MvPowerSeries.algebraMap_apply] }
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem coeToMvPowerSeries.algHom_apply :
    coeToMvPowerSeries.algHom A φ = MvPowerSeries.map σ (algebraMap R A) ↑φ :=
  rfl


instance algebraMvPolynomial : Algebra (MvPolynomial σ R) (MvPowerSeries σ A) :=
  RingHom.toAlgebra (MvPolynomial.coeToMvPowerSeries.algHom A).toRingHom


instance algebraMvPowerSeries : Algebra (MvPowerSeries σ R) (MvPowerSeries σ A) :=
  (map σ (algebraMap R A)).toAlgebra


theorem algebraMap_apply' (p : MvPolynomial σ R) :
    algebraMap (MvPolynomial σ R) (MvPowerSeries σ A) p = map σ (algebraMap R A) p :=
  rfl


theorem algebraMap_apply'' :
    algebraMap (MvPowerSeries σ R) (MvPowerSeries σ A) f = map σ (algebraMap R A) f :=
  rfl


