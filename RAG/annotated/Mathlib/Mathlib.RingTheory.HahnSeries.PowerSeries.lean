/-- The ring `HahnSeries ℕ R` is isomorphic to `PowerSeries R`. -/
@[simps]
def toPowerSeries : HahnSeries ℕ R ≃+* PowerSeries R where
  toFun f := PowerSeries.mk f.coeff
  invFun f := ⟨fun n => PowerSeries.coeff R n f, (Nat.lt_wfRel.wf.isWF _).isPWO⟩
  left_inv f := by
    /-
      Γ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      f : HahnSeries Nat R
      ⊢ Eq ((fun f => { coeff := fun n => (PowerSeries.coeff R n) f, isPWO_support'  …
    -/
    ext
    /-
      case coeff.h
      Γ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      f : HahnSeries Nat R
      x✝ : Nat
      ⊢ Eq (((fun f => { coeff := fun n => (PowerSeries.coeff R n) f, isPWO_support' …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      Γ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      f : PowerSeries R
      ⊢ Eq ((fun f => PowerSeries.mk f.coeff) ((fun f => { coeff := fun n => (PowerS …
    -/
    ext
    /-
      case h
      Γ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      f : PowerSeries R
      n✝ : Nat
      ⊢ Eq ((PowerSeries.coeff R n✝) ((fun f => PowerSeries.mk f.coeff) ((fun f => { …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_add' f g := by
    /-
      Γ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      f g : HahnSeries Nat R
      ⊢ Eq ({ toFun := fun f => PowerSeries.mk f.coeff, invFun := fun f => { coeff : …
    -/
    ext
    /-
      case h
      Γ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      f g : HahnSeries Nat R
      n✝ : Nat
      ⊢ Eq ((PowerSeries.coeff R n✝) ({ toFun := fun f => PowerSeries.mk f.coeff, in …
    -/
    /-
      Γ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      f g : HahnSeries Nat R
      ⊢ Eq ({ toFun := fun f => PowerSeries.mk f.coeff, invFun := fun f => { coeff : …
    -/
    simp
    /-
      case h
      Γ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      f g : HahnSeries Nat R
      n : Nat
      ⊢ Eq ((PowerSeries.coeff R n) ({ toFun := fun f => PowerSeries.mk f.coeff, inv …
    -/
    /-
      🎉 no goals
    -/
  map_mul' f g := by
    ext n
    simp only [PowerSeries.coeff_mul, PowerSeries.coeff_mk, mul_coeff, isPWO_support]
    classical
    refine (sum_filter_ne_zero _).symm.trans <| (sum_congr ?_ fun _ _ ↦ rfl).trans <|
      sum_filter_ne_zero _
    ext m
    simp only [mem_antidiagonal, mem_addAntidiagonal, and_congr_left_iff, mem_filter,
      mem_support]
    rintro h
    rw [and_iff_right (left_ne_zero_of_mul h), and_iff_right (right_ne_zero_of_mul h)]


theorem coeff_toPowerSeries {f : HahnSeries ℕ R} {n : ℕ} :
    PowerSeries.coeff R n (toPowerSeries f) = f.coeff n :=
  PowerSeries.coeff_mk _ _


theorem coeff_toPowerSeries_symm {f : PowerSeries R} {n : ℕ} :
    (HahnSeries.toPowerSeries.symm f).coeff n = PowerSeries.coeff R n f :=
  rfl


/-- Casts a power series as a Hahn series with coefficients from a `StrictOrderedSemiring`. -/
def ofPowerSeries : PowerSeries R →+* HahnSeries Γ R :=
  (HahnSeries.embDomainRingHom (Nat.castAddMonoidHom Γ) Nat.strictMono_cast.injective fun _ _ =>
        Nat.cast_le).comp
    (RingEquiv.toRingHom toPowerSeries.symm)


theorem ofPowerSeries_injective : Function.Injective (ofPowerSeries Γ R) :=
  embDomain_injective.comp toPowerSeries.symm.injective

/-@[simp] Porting note: removing simp. RHS is more complicated and it makes linter
failures elsewhere -/

theorem ofPowerSeries_apply (x : PowerSeries R) :
    ofPowerSeries Γ R x =
      HahnSeries.embDomain
        ⟨⟨((↑) : ℕ → Γ), Nat.strictMono_cast.injective⟩, by
          /-
            Γ : Type u_1
            R : Type u_2
            inst✝¹ : Semiring R
            inst✝ : StrictOrderedSemiring Γ
            x : PowerSeries R
            ⊢ ∀ {a b : Nat}, Iff (LE.le ({ toFun := Nat.cast, inj' := ⋯ } a) ({ toFun := N …
          -/
          simp only [Function.Embedding.coeFn_mk]
          /-
            Γ : Type u_1
            R : Type u_2
            inst✝¹ : Semiring R
            inst✝ : StrictOrderedSemiring Γ
            x : PowerSeries R
            ⊢ ∀ {a b : Nat}, Iff (LE.le ↑a ↑b) (LE.le a b)
          -/
          exact Nat.cast_le⟩
          /-
            🎉 no goals
          -/
        (toPowerSeries.symm x) :=
  rfl


theorem ofPowerSeries_apply_coeff (x : PowerSeries R) (n : ℕ) :
                                                                  /-
                                                                    Γ : Type u_1
                                                                    R : Type u_2
                                                                    inst✝¹ : Semiring R
                                                                    inst✝ : StrictOrderedSemiring Γ
                                                                    x : PowerSeries R
                                                                    n : Nat
                                                                    ⊢ Eq (((HahnSeries.ofPowerSeries Γ R) x).coeff ↑n) ((PowerSeries.coeff R n) x)
                                                                  -/
    (ofPowerSeries Γ R x).coeff n = PowerSeries.coeff R n x := by simp [ofPowerSeries_apply]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem ofPowerSeries_C (r : R) : ofPowerSeries Γ R (PowerSeries.C R r) = HahnSeries.C r := by
  /-
    Γ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : StrictOrderedSemiring Γ
    r : R
    ⊢ Eq ((HahnSeries.ofPowerSeries Γ R) ((PowerSeries.C R) r)) (HahnSeries.C r)
  -/
  ext n
  simp only [ofPowerSeries_apply, C, RingHom.coe_mk, MonoidHom.coe_mk, OneHom.coe_mk, ne_eq,
    single_coeff]
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : StrictOrderedSemiring Γ
    r : R
    n : Γ
    ⊢ Eq ((HahnSeries.embDomain { toFun := Nat.cast, inj' := ⋯, map_rel_iff' := ⋯  …
  -/
  split_ifs with hn
    /-
      case pos
      Γ : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : StrictOrderedSemiring Γ
      r : R
      n : Γ
      hn : Eq n 0
      ⊢ Eq ((HahnSeries.embDomain { toFun := Nat.cast, inj' := ⋯, map_rel_iff' := ⋯  …
    -/
  · subst hn
    /-
      case pos
      Γ : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : StrictOrderedSemiring Γ
      r : R
      ⊢ Eq ((HahnSeries.embDomain { toFun := Nat.cast, inj' := ⋯, map_rel_iff' := ⋯  …
    -/
                                         /-
                                           🎉 no goals
                                         -/
    convert embDomain_coeff (a := 0) <;> simp
                                         /-
                                           🎉 no goals
                                         -/
    /-
      case neg
      Γ : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : StrictOrderedSemiring Γ
      r : R
      n : Γ
      hn : Not (Eq n 0)
      ⊢ Eq ((HahnSeries.embDomain { toFun := Nat.cast, inj' := ⋯, map_rel_iff' := ⋯  …
    -/
  · rw [embDomain_notin_image_support]
    simp only [not_exists, Set.mem_image, toPowerSeries_symm_apply_coeff, mem_support,
      PowerSeries.coeff_C]
    /-
      case neg
      Γ : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : StrictOrderedSemiring Γ
      r : R
      n : Γ
      hn : Not (Eq n 0)
      ⊢ ∀ (x : Nat), Not (And (Ne (ite (Eq x 0) r 0) 0) (Eq ({ toFun := Nat.cast, in …
    -/
    intro
    /-
      case neg
      Γ : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : StrictOrderedSemiring Γ
      r : R
      n : Γ
      hn : Not (Eq n 0)
      x✝ : Nat
      ⊢ Not (And (Ne (ite (Eq x✝ 0) r 0) 0) (Eq ({ toFun := Nat.cast, inj' := ⋯, map …
    -/
    simp +contextual [Ne.symm hn]
    /-
      🎉 no goals
    -/


@[simp]
theorem ofPowerSeries_X : ofPowerSeries Γ R PowerSeries.X = single 1 1 := by
  /-
    Γ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : StrictOrderedSemiring Γ
    ⊢ Eq ((HahnSeries.ofPowerSeries Γ R) PowerSeries.X) ((HahnSeries.single 1) 1)
  -/
  ext n
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : StrictOrderedSemiring Γ
    n : Γ
    ⊢ Eq (((HahnSeries.ofPowerSeries Γ R) PowerSeries.X).coeff n) (((HahnSeries.si …
  -/
  simp only [single_coeff, ofPowerSeries_apply, RingHom.coe_mk]
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : StrictOrderedSemiring Γ
    n : Γ
    ⊢ Eq ((HahnSeries.embDomain { toFun := Nat.cast, inj' := ⋯, map_rel_iff' := ⋯  …
  -/
  split_ifs with hn
    /-
      case pos
      Γ : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : StrictOrderedSemiring Γ
      n : Γ
      hn : Eq n 1
      ⊢ Eq ((HahnSeries.embDomain { toFun := Nat.cast, inj' := ⋯, map_rel_iff' := ⋯  …
    -/
  · rw [hn]
    /-
      case pos
      Γ : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : StrictOrderedSemiring Γ
      n : Γ
      hn : Eq n 1
      ⊢ Eq ((HahnSeries.embDomain { toFun := Nat.cast, inj' := ⋯, map_rel_iff' := ⋯  …
    -/
                                         /-
                                           🎉 no goals
                                         -/
    convert embDomain_coeff (a := 1) <;> simp
                                         /-
                                           🎉 no goals
                                         -/
    /-
      case neg
      Γ : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : StrictOrderedSemiring Γ
      n : Γ
      hn : Not (Eq n 1)
      ⊢ Eq ((HahnSeries.embDomain { toFun := Nat.cast, inj' := ⋯, map_rel_iff' := ⋯  …
    -/
  · rw [embDomain_notin_image_support]
    simp only [not_exists, Set.mem_image, toPowerSeries_symm_apply_coeff, mem_support,
      PowerSeries.coeff_X]
    /-
      case neg
      Γ : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : StrictOrderedSemiring Γ
      n : Γ
      hn : Not (Eq n 1)
      ⊢ ∀ (x : Nat), Not (And (Ne (ite (Eq x 1) 1 0) 0) (Eq ({ toFun := Nat.cast, in …
    -/
    intro
    /-
      case neg
      Γ : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : StrictOrderedSemiring Γ
      n : Γ
      hn : Not (Eq n 1)
      x✝ : Nat
      ⊢ Not (And (Ne (ite (Eq x✝ 1) 1 0) 0) (Eq ({ toFun := Nat.cast, inj' := ⋯, map …
    -/
    simp +contextual [Ne.symm hn]
    /-
      🎉 no goals
    -/


theorem ofPowerSeries_X_pow {R} [Semiring R] (n : ℕ) :
    ofPowerSeries Γ R (PowerSeries.X ^ n) = single (n : Γ) 1 := by
  /-
    Γ : Type u_1
    inst✝¹ : StrictOrderedSemiring Γ
    R : Type u_3
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq ((HahnSeries.ofPowerSeries Γ R) (HPow.hPow PowerSeries.X n)) ((HahnSeries …
  -/
  simp
  /-
    🎉 no goals
  -/

-- Lemmas about converting hahn_series over fintype to and from mv_power_series

/-- The ring `HahnSeries (σ →₀ ℕ) R` is isomorphic to `MvPowerSeries σ R` for a `Finite` `σ`.
We take the index set of the hahn series to be `Finsupp` rather than `pi`,
even though we assume `Finite σ` as this is more natural for alignment with `MvPowerSeries`.
After importing `Algebra.Order.Pi` the ring `HahnSeries (σ → ℕ) R` could be constructed instead.
 -/
@[simps]
def toMvPowerSeries {σ : Type*} [Finite σ] : HahnSeries (σ →₀ ℕ) R ≃+* MvPowerSeries σ R where
  toFun f := f.coeff
  invFun f := ⟨(f : (σ →₀ ℕ) → R), Finsupp.isPWO _⟩
  left_inv f := by
    /-
      Γ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : StrictOrderedSemiring Γ
      σ : Type u_3
      inst✝ : Finite σ
      f : HahnSeries (Finsupp σ Nat) R
      ⊢ Eq ((fun f => { coeff := f, isPWO_support' := ⋯ }) ((fun f => f.coeff) f)) f
    -/
    ext
    /-
      case coeff.h
      Γ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : StrictOrderedSemiring Γ
      σ : Type u_3
      inst✝ : Finite σ
      f : HahnSeries (Finsupp σ Nat) R
      x✝ : Finsupp σ Nat
      ⊢ Eq (((fun f => { coeff := f, isPWO_support' := ⋯ }) ((fun f => f.coeff) f)). …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      Γ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : StrictOrderedSemiring Γ
      σ : Type u_3
      inst✝ : Finite σ
      f : MvPowerSeries σ R
      ⊢ Eq ((fun f => f.coeff) ((fun f => { coeff := f, isPWO_support' := ⋯ }) f)) f
    -/
    ext
    /-
      case h
      Γ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : StrictOrderedSemiring Γ
      σ : Type u_3
      inst✝ : Finite σ
      f : MvPowerSeries σ R
      n✝ : Finsupp σ Nat
      ⊢ Eq ((MvPowerSeries.coeff R n✝) ((fun f => f.coeff) ((fun f => { coeff := f,  …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_add' f g := by
    /-
      Γ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : StrictOrderedSemiring Γ
      σ : Type u_3
      inst✝ : Finite σ
      f g : HahnSeries (Finsupp σ Nat) R
      ⊢ Eq ({ toFun := fun f => f.coeff, invFun := fun f => { coeff := f, isPWO_supp …
    -/
    ext
    /-
      case h
      Γ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : StrictOrderedSemiring Γ
      σ : Type u_3
      inst✝ : Finite σ
      f g : HahnSeries (Finsupp σ Nat) R
      n✝ : Finsupp σ Nat
      ⊢ Eq ((MvPowerSeries.coeff R n✝) ({ toFun := fun f => f.coeff, invFun := fun f …
    -/
    /-
      Γ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : StrictOrderedSemiring Γ
      σ : Type u_3
      inst✝ : Finite σ
      f g : HahnSeries (Finsupp σ Nat) R
      ⊢ Eq ({ toFun := fun f => f.coeff, invFun := fun f => { coeff := f, isPWO_supp …
    -/
    simp
    /-
      case h
      Γ : Type u_1
      R : Type u_2
      inst✝² : Semiring R
      inst✝¹ : StrictOrderedSemiring Γ
      σ : Type u_3
      inst✝ : Finite σ
      f g : HahnSeries (Finsupp σ Nat) R
      n : Finsupp σ Nat
      ⊢ Eq ((MvPowerSeries.coeff R n) ({ toFun := fun f => f.coeff, invFun := fun f  …
    -/
    /-
      🎉 no goals
    -/
  map_mul' f g := by
    ext n
    simp only [MvPowerSeries.coeff_mul]
    classical
      change (f * g).coeff n = _
      simp_rw [mul_coeff]
      refine (sum_filter_ne_zero _).symm.trans <| (sum_congr ?_ fun _ _ ↦ rfl).trans <|
        sum_filter_ne_zero _
      ext m
      simp only [and_congr_left_iff, mem_addAntidiagonal, mem_filter, mem_support,
        Finset.mem_antidiagonal]
      rintro h
      rw [and_iff_right (left_ne_zero_of_mul h), and_iff_right (right_ne_zero_of_mul h)]


/-- If R has no zero divisors and `σ` is finite,
then `HahnSeries (σ →₀ ℕ) R` has no zero divisors -/
instance [NoZeroDivisors R] : NoZeroDivisors (HahnSeries (σ →₀ ℕ) R) :=
  toMvPowerSeries.toMulEquiv.noZeroDivisors (A := HahnSeries (σ →₀ ℕ) R) (MvPowerSeries σ R)


theorem coeff_toMvPowerSeries {f : HahnSeries (σ →₀ ℕ) R} {n : σ →₀ ℕ} :
    MvPowerSeries.coeff R n (toMvPowerSeries f) = f.coeff n :=
  rfl


theorem coeff_toMvPowerSeries_symm {f : MvPowerSeries σ R} {n : σ →₀ ℕ} :
    (HahnSeries.toMvPowerSeries.symm f).coeff n = MvPowerSeries.coeff R n f :=
  rfl


/-- The `R`-algebra `HahnSeries ℕ A` is isomorphic to `PowerSeries A`. -/
@[simps!]
def toPowerSeriesAlg : HahnSeries ℕ A ≃ₐ[R] PowerSeries A :=
  { toPowerSeries with
    commutes' := fun r => by
      /-
        Γ : Type u_1
        R : Type u_2
        inst✝² : CommSemiring R
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        r : R
        ⊢ Eq (__src✝.toFun ((algebraMap R (HahnSeries Nat A)) r)) ((algebraMap R (Powe …
      -/
      ext n
      /-
        case h
        Γ : Type u_1
        R : Type u_2
        inst✝² : CommSemiring R
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        r : R
        n : Nat
        ⊢ Eq ((PowerSeries.coeff A n) (__src✝.toFun ((algebraMap R (HahnSeries Nat A)) …
      -/
                  /-
                    🎉 no goals
                  -/
      cases n <;> simp [algebraMap_apply, PowerSeries.algebraMap_apply] }
                  /-
                    🎉 no goals
                  -/


/-- Casting a power series as a Hahn series with coefficients from a `StrictOrderedSemiring`
  is an algebra homomorphism. -/
@[simps!]
def ofPowerSeriesAlg : PowerSeries A →ₐ[R] HahnSeries Γ A :=
  (HahnSeries.embDomainAlgHom (Nat.castAddMonoidHom Γ) Nat.strictMono_cast.injective fun _ _ =>
        Nat.cast_le).comp
    (AlgEquiv.toAlgHom (toPowerSeriesAlg R).symm)


instance powerSeriesAlgebra {S : Type*} [CommSemiring S] [Algebra S (PowerSeries R)] :
    Algebra S (HahnSeries Γ R) :=
  RingHom.toAlgebra <| (ofPowerSeries Γ R).comp (algebraMap S (PowerSeries R))


theorem algebraMap_apply' (x : S) :
    algebraMap S (HahnSeries Γ R) x = ofPowerSeries Γ R (algebraMap S (PowerSeries R) x) :=
  rfl


@[simp]
theorem _root_.Polynomial.algebraMap_hahnSeries_apply (f : R[X]) :
    algebraMap R[X] (HahnSeries Γ R) f = ofPowerSeries Γ R f :=
  rfl


theorem _root_.Polynomial.algebraMap_hahnSeries_injective :
    Function.Injective (algebraMap R[X] (HahnSeries Γ R)) :=
  ofPowerSeries_injective.comp (Polynomial.coe_injective R)


