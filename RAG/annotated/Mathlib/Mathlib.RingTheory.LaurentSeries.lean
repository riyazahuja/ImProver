/-- `LaurentSeries R` is the type of formal Laurent series with coefficients in `R`, denoted `R⸨X⸩`.

  It is implemented as a `HahnSeries` with value group `ℤ`.
-/
abbrev LaurentSeries (R : Type u) [Zero R] :=
  HahnSeries ℤ R


/--
`R⸨X⸩` is notation for `LaurentSeries R`,
-/
scoped notation:9000 R "⸨X⸩" => LaurentSeries R


/-- The Hasse derivative of Laurent series, as a linear map. -/
def hasseDeriv (R : Type*) {V : Type*} [AddCommGroup V] [Semiring R] [Module R V] (k : ℕ) :
    V⸨X⸩ →ₗ[R] V⸨X⸩ where
  toFun f := HahnSeries.ofSuppBddBelow (fun (n : ℤ) => (Ring.choose (n + k) k) • f.coeff (n + k))
    (forallLTEqZero_supp_BddBelow _ (f.order - k : ℤ)
                     /-
                       R✝ : Type u_1
                       R : Type u_2
                       V : Type u_3
                       inst✝² : AddCommGroup V
                       inst✝¹ : Semiring R
                       inst✝ : Module R V
                       k : Nat
                       f : LaurentSeries V
                       x✝ : Int
                       h_lt : LT.lt x✝ (HSub.hSub (HahnSeries.order f) ↑k)
                       ⊢ Eq (HSMul.hSMul (Ring.choose (HAdd.hAdd x✝ ↑k) k) (f.coeff (HAdd.hAdd x✝ ↑k) …
                     -/
    (fun _ h_lt ↦ by rw [coeff_eq_zero_of_lt_order <| lt_sub_iff_add_lt.mp h_lt, smul_zero]))
                     /-
                       🎉 no goals
                     -/
  map_add' f g := by
    /-
      R✝ : Type u_1
      R : Type u_2
      V : Type u_3
      inst✝² : AddCommGroup V
      inst✝¹ : Semiring R
      inst✝ : Module R V
      k : Nat
      f g : LaurentSeries V
      ⊢ Eq ((fun f => HahnSeries.ofSuppBddBelow (fun n => HSMul.hSMul (Ring.choose ( …
    -/
    ext
    /-
      case coeff.h
      R✝ : Type u_1
      R : Type u_2
      V : Type u_3
      inst✝² : AddCommGroup V
      inst✝¹ : Semiring R
      inst✝ : Module R V
      k : Nat
      f g : LaurentSeries V
      x✝ : Int
      ⊢ Eq (((fun f => HahnSeries.ofSuppBddBelow (fun n => HSMul.hSMul (Ring.choose  …
    -/
    simp only [ofSuppBddBelow, add_coeff', Pi.add_apply, smul_add]
    /-
      🎉 no goals
    -/
  map_smul' r f := by
    /-
      R✝ : Type u_1
      R : Type u_2
      V : Type u_3
      inst✝² : AddCommGroup V
      inst✝¹ : Semiring R
      inst✝ : Module R V
      k : Nat
      r : R
      f : LaurentSeries V
      ⊢ Eq ({ toFun := fun f => HahnSeries.ofSuppBddBelow (fun n => HSMul.hSMul (Rin …
    -/
    ext
    /-
      case coeff.h
      R✝ : Type u_1
      R : Type u_2
      V : Type u_3
      inst✝² : AddCommGroup V
      inst✝¹ : Semiring R
      inst✝ : Module R V
      k : Nat
      r : R
      f : LaurentSeries V
      x✝ : Int
      ⊢ Eq (({ toFun := fun f => HahnSeries.ofSuppBddBelow (fun n => HSMul.hSMul (Ri …
    -/
    simp only [ofSuppBddBelow, smul_coeff, RingHom.id_apply, smul_comm r]
    /-
      🎉 no goals
    -/


@[simp]
theorem hasseDeriv_coeff (k : ℕ) (f : LaurentSeries V) (n : ℤ) :
    (hasseDeriv R k f).coeff n = Ring.choose (n + k) k • f.coeff (n + k) :=
  rfl


@[simp]
theorem hasseDeriv_zero : hasseDeriv R 0 = LinearMap.id (M := LaurentSeries V) := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    V : Type u_2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    ⊢ Eq (LaurentSeries.hasseDeriv R 0) LinearMap.id
  -/
  ext f n
  /-
    case h.coeff.h
    R : Type u_1
    inst✝² : Semiring R
    V : Type u_2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    f : LaurentSeries V
    n : Int
    ⊢ Eq (((LaurentSeries.hasseDeriv R 0) f).coeff n) ((LinearMap.id f).coeff n)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem hasseDeriv_single_add (k : ℕ) (n : ℤ) (x : V) :
    hasseDeriv R k (single (n + k) x) = single n ((Ring.choose (n + k) k) • x) := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    V : Type u_2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    k : Nat
    n : Int
    x : V
    ⊢ Eq ((LaurentSeries.hasseDeriv R k) ((HahnSeries.single (HAdd.hAdd n ↑k)) x)) …
  -/
  ext m
  /-
    case coeff.h
    R : Type u_1
    inst✝² : Semiring R
    V : Type u_2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    k : Nat
    n : Int
    x : V
    m : Int
    ⊢ Eq (((LaurentSeries.hasseDeriv R k) ((HahnSeries.single (HAdd.hAdd n ↑k)) x) …
  -/
  dsimp only [hasseDeriv_coeff]
  /-
    case coeff.h
    R : Type u_1
    inst✝² : Semiring R
    V : Type u_2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    k : Nat
    n : Int
    x : V
    m : Int
    ⊢ Eq (HSMul.hSMul (Ring.choose (HAdd.hAdd m ↑k) k) (((HahnSeries.single (HAdd. …
  -/
  by_cases h : m = n
    /-
      case pos
      R : Type u_1
      inst✝² : Semiring R
      V : Type u_2
      inst✝¹ : AddCommGroup V
      inst✝ : Module R V
      k : Nat
      n : Int
      x : V
      m : Int
      h : Eq m n
      ⊢ Eq (HSMul.hSMul (Ring.choose (HAdd.hAdd m ↑k) k) (((HahnSeries.single (HAdd. …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝² : Semiring R
      V : Type u_2
      inst✝¹ : AddCommGroup V
      inst✝ : Module R V
      k : Nat
      n : Int
      x : V
      m : Int
      h : Not (Eq m n)
      ⊢ Eq (HSMul.hSMul (Ring.choose (HAdd.hAdd m ↑k) k) (((HahnSeries.single (HAdd. …
    -/
  · simp [h, show m + k ≠ n + k by omega]
    /-
      🎉 no goals
    -/


@[simp]
theorem hasseDeriv_single (k : ℕ) (n : ℤ) (x : V) :
    hasseDeriv R k (single n x) = single (n - k) ((Ring.choose n k) • x) := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    V : Type u_2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    k : Nat
    n : Int
    x : V
    ⊢ Eq ((LaurentSeries.hasseDeriv R k) ((HahnSeries.single n) x)) ((HahnSeries.s …
  -/
  rw [← Int.sub_add_cancel n k, hasseDeriv_single_add, Int.sub_add_cancel n k]
  /-
    🎉 no goals
  -/


theorem hasseDeriv_comp_coeff (k l : ℕ) (f : LaurentSeries V) (n : ℤ) :
    (hasseDeriv R k (hasseDeriv R l f)).coeff n =
      ((Nat.choose (k + l) k) • hasseDeriv R (k + l) f).coeff n := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    V : Type u_2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    k l : Nat
    f : LaurentSeries V
    n : Int
    ⊢ Eq (((LaurentSeries.hasseDeriv R k) ((LaurentSeries.hasseDeriv R l) f)).coef …
  -/
  rw [nsmul_coeff]
  /-
    R : Type u_1
    inst✝² : Semiring R
    V : Type u_2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    k l : Nat
    f : LaurentSeries V
    n : Int
    ⊢ Eq (((LaurentSeries.hasseDeriv R k) ((LaurentSeries.hasseDeriv R l) f)).coef …
  -/
  simp only [hasseDeriv_coeff, Pi.smul_apply, Nat.cast_add]
  rw [smul_smul, mul_comm, ← Ring.choose_add_smul_choose (n + k), add_assoc, Nat.choose_symm_add,
    smul_assoc]


@[simp]
theorem hasseDeriv_comp (k l : ℕ) (f : LaurentSeries V) :
    hasseDeriv R k (hasseDeriv R l f) = (k + l).choose k • hasseDeriv R (k + l) f := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    V : Type u_2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    k l : Nat
    f : LaurentSeries V
    ⊢ Eq ((LaurentSeries.hasseDeriv R k) ((LaurentSeries.hasseDeriv R l) f)) (HSMu …
  -/
  ext n
  /-
    case coeff.h
    R : Type u_1
    inst✝² : Semiring R
    V : Type u_2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    k l : Nat
    f : LaurentSeries V
    n : Int
    ⊢ Eq (((LaurentSeries.hasseDeriv R k) ((LaurentSeries.hasseDeriv R l) f)).coef …
  -/
  simp [hasseDeriv_comp_coeff k l f n]
  /-
    🎉 no goals
  -/


/-- The derivative of a Laurent series. -/
def derivative (R : Type*) {V : Type*} [AddCommGroup V] [Semiring R] [Module R V] :
    LaurentSeries V →ₗ[R] LaurentSeries V :=
  hasseDeriv R 1


@[simp]
theorem derivative_apply (f : LaurentSeries V) : derivative R f = hasseDeriv R 1 f := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    V : Type u_2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    f : LaurentSeries V
    ⊢ Eq ((LaurentSeries.derivative R) f) ((LaurentSeries.hasseDeriv R 1) f)
  -/
  exact rfl
  /-
    🎉 no goals
  -/


theorem derivative_iterate (k : ℕ) (f : LaurentSeries V) :
    (derivative R)^[k] f = k.factorial • (hasseDeriv R k f) := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    V : Type u_2
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    k : Nat
    f : LaurentSeries V
    ⊢ Eq (Nat.iterate (⇑(LaurentSeries.derivative R)) k f) (HSMul.hSMul k.factoria …
  -/
  ext n
  induction k generalizing f with
  | zero => simp
  | succ k ih =>
    rw [Function.iterate_succ, Function.comp_apply, ih, derivative_apply, hasseDeriv_comp,
      Nat.choose_symm_add, Nat.choose_one_right, Nat.factorial, mul_nsmul]


@[simp]
theorem derivative_iterate_coeff (k : ℕ) (f : LaurentSeries V) (n : ℤ) :
    ((derivative R)^[k] f).coeff n = (descPochhammer ℤ k).smeval (n + k) • f.coeff (n + k) := by
  rw [derivative_iterate, nsmul_coeff, Pi.smul_apply, hasseDeriv_coeff,
    Ring.descPochhammer_eq_factorial_smul_choose, smul_assoc]


instance : Coe R⟦X⟧ R⸨X⸩ :=
  ⟨HahnSeries.ofPowerSeries ℤ R⟩

/- Porting note: now a syntactic tautology and not needed elsewhere
theorem coe_powerSeries (x : R⟦X⟧) :
    (x : R⸨X⸩) = HahnSeries.ofPowerSeries ℤ R x :=
  rfl -/


@[simp]
theorem coeff_coe_powerSeries (x : R⟦X⟧) (n : ℕ) :
    HahnSeries.coeff (x : R⸨X⸩) n = PowerSeries.coeff R n x := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x : PowerSeries R
    n : Nat
    ⊢ Eq (((HahnSeries.ofPowerSeries Int R) x).coeff ↑n) ((PowerSeries.coeff R n) x)
  -/
  rw [ofPowerSeries_apply_coeff]
  /-
    🎉 no goals
  -/


/-- This is a power series that can be multiplied by an integer power of `X` to give our
  Laurent series. If the Laurent series is nonzero, `powerSeriesPart` has a nonzero
  constant term. -/
def powerSeriesPart (x : R⸨X⸩) : R⟦X⟧ :=
  PowerSeries.mk fun n => x.coeff (x.order + n)


@[simp]
theorem powerSeriesPart_coeff (x : R⸨X⸩) (n : ℕ) :
    PowerSeries.coeff R n x.powerSeriesPart = x.coeff (x.order + n) :=
  PowerSeries.coeff_mk _ _


@[simp]
theorem powerSeriesPart_zero : powerSeriesPart (0 : R⸨X⸩) = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Eq (LaurentSeries.powerSeriesPart 0) 0
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    n✝ : Nat
    ⊢ Eq ((PowerSeries.coeff R n✝) (LaurentSeries.powerSeriesPart 0)) ((PowerSerie …
  -/
  simp [(PowerSeries.coeff _ _).map_zero] -- Note: this doesn't get picked up any more
  /-
    🎉 no goals
  -/


@[simp]
theorem powerSeriesPart_eq_zero (x : R⸨X⸩) : x.powerSeriesPart = 0 ↔ x = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x : LaurentSeries R
    ⊢ Iff (Eq x.powerSeriesPart 0) (Eq x 0)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : Semiring R
      x : LaurentSeries R
      ⊢ Eq x.powerSeriesPart 0 → Eq x 0
    -/
  · contrapose!
    /-
      case mp
      R : Type u_1
      inst✝ : Semiring R
      x : LaurentSeries R
      ⊢ Ne x 0 → Ne x.powerSeriesPart 0
    -/
    simp only [ne_eq]
    /-
      case mp
      R : Type u_1
      inst✝ : Semiring R
      x : LaurentSeries R
      ⊢ Not (Eq x 0) → Not (Eq x.powerSeriesPart 0)
    -/
    intro h
    /-
      case mp
      R : Type u_1
      inst✝ : Semiring R
      x : LaurentSeries R
      h : Not (Eq x 0)
      ⊢ Not (Eq x.powerSeriesPart 0)
    -/
    rw [PowerSeries.ext_iff, not_forall]
    /-
      case mp
      R : Type u_1
      inst✝ : Semiring R
      x : LaurentSeries R
      h : Not (Eq x 0)
      ⊢ Exists fun x_1 => Not (Eq ((PowerSeries.coeff R x_1) x.powerSeriesPart) ((Po …
    -/
    refine ⟨0, ?_⟩
    /-
      case mp
      R : Type u_1
      inst✝ : Semiring R
      x : LaurentSeries R
      h : Not (Eq x 0)
      ⊢ Not (Eq ((PowerSeries.coeff R 0) x.powerSeriesPart) ((PowerSeries.coeff R 0) …
    -/
    simp [coeff_order_ne_zero h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝ : Semiring R
      x : LaurentSeries R
      ⊢ Eq x 0 → Eq x.powerSeriesPart 0
    -/
  · rintro rfl
    /-
      case mpr
      R : Type u_1
      inst✝ : Semiring R
      ⊢ Eq (LaurentSeries.powerSeriesPart 0) 0
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem single_order_mul_powerSeriesPart (x : R⸨X⸩) :
    (single x.order 1 : R⸨X⸩) * x.powerSeriesPart = x := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x : LaurentSeries R
    ⊢ Eq (HMul.hMul ((HahnSeries.single (HahnSeries.order x)) 1) ((HahnSeries.ofPo …
  -/
  ext n
  /-
    case coeff.h
    R : Type u_1
    inst✝ : Semiring R
    x : LaurentSeries R
    n : Int
    ⊢ Eq ((HMul.hMul ((HahnSeries.single (HahnSeries.order x)) 1) ((HahnSeries.ofP …
  -/
  rw [← sub_add_cancel n x.order, single_mul_coeff_add, sub_add_cancel, one_mul]
  /-
    case coeff.h
    R : Type u_1
    inst✝ : Semiring R
    x : LaurentSeries R
    n : Int
    ⊢ Eq (((HahnSeries.ofPowerSeries Int R) x.powerSeriesPart).coeff (HSub.hSub n  …
  -/
  by_cases h : x.order ≤ n
  · rw [Int.eq_natAbs_of_zero_le (sub_nonneg_of_le h), coeff_coe_powerSeries,
      powerSeriesPart_coeff, ← Int.eq_natAbs_of_zero_le (sub_nonneg_of_le h),
      add_sub_cancel]
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      x : LaurentSeries R
      n : Int
      h : Not (LE.le (HahnSeries.order x) n)
      ⊢ Eq (((HahnSeries.ofPowerSeries Int R) x.powerSeriesPart).coeff (HSub.hSub n  …
    -/
  · rw [ofPowerSeries_apply, embDomain_notin_range]
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        x : LaurentSeries R
        n : Int
        h : Not (LE.le (HahnSeries.order x) n)
        ⊢ Eq 0 (x.coeff n)
      -/
    · contrapose! h
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        x : LaurentSeries R
        n : Int
        h : Ne 0 (x.coeff n)
        ⊢ LE.le (HahnSeries.order x) n
      -/
      exact order_le_of_coeff_ne_zero h.symm
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        x : LaurentSeries R
        n : Int
        h : Not (LE.le (HahnSeries.order x) n)
        ⊢ Not (Membership.mem (Set.range ⇑{ toFun := Nat.cast, inj' := ⋯, map_rel_iff' …
      -/
    · contrapose! h
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        x : LaurentSeries R
        n : Int
        h : Membership.mem (Set.range ⇑{ toFun := Nat.cast, inj' := ⋯, map_rel_iff' := …
        ⊢ LE.le (HahnSeries.order x) n
      -/
      simp only [Set.mem_range, RelEmbedding.coe_mk, Function.Embedding.coeFn_mk] at h
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        x : LaurentSeries R
        n : Int
        h : Exists fun y => Eq (↑y) (HSub.hSub n (HahnSeries.order x))
        ⊢ LE.le (HahnSeries.order x) n
      -/
      obtain ⟨m, hm⟩ := h
      /-
        case neg.intro
        R : Type u_1
        inst✝ : Semiring R
        x : LaurentSeries R
        n : Int
        m : Nat
        hm : Eq (↑m) (HSub.hSub n (HahnSeries.order x))
        ⊢ LE.le (HahnSeries.order x) n
      -/
      rw [← sub_nonneg, ← hm]
      /-
        case neg.intro
        R : Type u_1
        inst✝ : Semiring R
        x : LaurentSeries R
        n : Int
        m : Nat
        hm : Eq (↑m) (HSub.hSub n (HahnSeries.order x))
        ⊢ LE.le 0 ↑m
      -/
      simp only [Nat.cast_nonneg]
      /-
        🎉 no goals
      -/


theorem ofPowerSeries_powerSeriesPart (x : R⸨X⸩) :
    ofPowerSeries ℤ R x.powerSeriesPart = single (-x.order) 1 * x := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x : LaurentSeries R
    ⊢ Eq ((HahnSeries.ofPowerSeries Int R) x.powerSeriesPart) (HMul.hMul ((HahnSer …
  -/
  refine Eq.trans ?_ (congr rfl x.single_order_mul_powerSeriesPart)
  /-
    R : Type u_1
    inst✝ : Semiring R
    x : LaurentSeries R
    ⊢ Eq ((HahnSeries.ofPowerSeries Int R) x.powerSeriesPart) (HMul.hMul ((HahnSer …
  -/
  rw [← mul_assoc, single_mul_single, neg_add_cancel, mul_one, ← C_apply, C_one, one_mul]
  /-
    🎉 no goals
  -/


theorem X_order_mul_powerSeriesPart {n : ℕ} {f : R⸨X⸩} (hn : n = f.order) :
    (PowerSeries.X ^ n * f.powerSeriesPart : R⟦X⟧) = f := by
  simp only [map_mul, map_pow, ofPowerSeries_X, single_pow, nsmul_eq_mul, mul_one, one_pow, hn,
    single_order_mul_powerSeriesPart]


instance [CommSemiring R] : Algebra R⟦X⟧ R⸨X⸩ := (HahnSeries.ofPowerSeries ℤ R).toAlgebra


@[simp]
theorem coe_algebraMap [CommSemiring R] :
    ⇑(algebraMap R⟦X⟧ R⸨X⸩) = HahnSeries.ofPowerSeries ℤ R :=
  rfl


/-- The localization map from power series to Laurent series. -/
@[simps (config := { rhsMd := .all, simpRhs := true })]
instance of_powerSeries_localization [CommRing R] :
    IsLocalization (Submonoid.powers (PowerSeries.X : R⟦X⟧)) R⸨X⸩ where
  map_units' := by
    /-
      R : Type u_1
      inst✝ : CommRing R
      ⊢ ∀ (y : Subtype fun x => Membership.mem (Submonoid.powers PowerSeries.X) x),  …
    -/
    rintro ⟨_, n, rfl⟩
    /-
      case mk.intro
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ⊢ IsUnit ((algebraMap (PowerSeries R) (LaurentSeries R)) ↑⟨(fun x => HPow.hPow …
    -/
    refine ⟨⟨single (n : ℤ) 1, single (-n : ℤ) 1, ?_, ?_⟩, ?_⟩
      /-
        case mk.intro.refine_1
        R : Type u_1
        inst✝ : CommRing R
        n : Nat
        ⊢ Eq (HMul.hMul ((HahnSeries.single ↑n) 1) ((HahnSeries.single (Neg.neg ↑n)) 1 …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mk.intro.refine_2
        R : Type u_1
        inst✝ : CommRing R
        n : Nat
        ⊢ Eq (HMul.hMul ((HahnSeries.single (Neg.neg ↑n)) 1) ((HahnSeries.single ↑n) 1 …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mk.intro.refine_3
        R : Type u_1
        inst✝ : CommRing R
        n : Nat
        ⊢ Eq (↑{ val := (HahnSeries.single ↑n) 1, inv := (HahnSeries.single (Neg.neg ↑ …
      -/
    · dsimp; rw [ofPowerSeries_X_pow]
             /-
               🎉 no goals
             -/
  surj' z := by
    /-
      R : Type u_1
      inst✝ : CommRing R
      z : LaurentSeries R
      ⊢ Exists fun x => Eq (HMul.hMul z ((algebraMap (PowerSeries R) (LaurentSeries  …
    -/
    by_cases h : 0 ≤ z.order
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        z : LaurentSeries R
        h : LE.le 0 (HahnSeries.order z)
        ⊢ Exists fun x => Eq (HMul.hMul z ((algebraMap (PowerSeries R) (LaurentSeries  …
      -/
    · refine ⟨⟨PowerSeries.X ^ Int.natAbs z.order * powerSeriesPart z, 1⟩, ?_⟩
      simp only [RingHom.map_one, mul_one, RingHom.map_mul, coe_algebraMap, ofPowerSeries_X_pow,
        Submonoid.coe_one]
      /-
        case pos
        R : Type u_1
        inst✝ : CommRing R
        z : LaurentSeries R
        h : LE.le 0 (HahnSeries.order z)
        ⊢ Eq z (HMul.hMul ((HahnSeries.single ↑(HahnSeries.order z).natAbs) 1) ((HahnS …
      -/
      rw [Int.natAbs_of_nonneg h, single_order_mul_powerSeriesPart]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        z : LaurentSeries R
        h : Not (LE.le 0 (HahnSeries.order z))
        ⊢ Exists fun x => Eq (HMul.hMul z ((algebraMap (PowerSeries R) (LaurentSeries  …
      -/
    · refine ⟨⟨powerSeriesPart z, PowerSeries.X ^ Int.natAbs z.order, ⟨_, rfl⟩⟩, ?_⟩
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        z : LaurentSeries R
        h : Not (LE.le 0 (HahnSeries.order z))
        ⊢ Eq (HMul.hMul z ((algebraMap (PowerSeries R) (LaurentSeries R)) ↑{ fst := z. …
      -/
      simp only [coe_algebraMap, ofPowerSeries_powerSeriesPart]
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        z : LaurentSeries R
        h : Not (LE.le 0 (HahnSeries.order z))
        ⊢ Eq (HMul.hMul z ((HahnSeries.ofPowerSeries Int R) (HPow.hPow PowerSeries.X ( …
      -/
      rw [mul_comm _ z]
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        z : LaurentSeries R
        h : Not (LE.le 0 (HahnSeries.order z))
        ⊢ Eq (HMul.hMul z ((HahnSeries.ofPowerSeries Int R) (HPow.hPow PowerSeries.X ( …
      -/
      refine congr rfl ?_
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        z : LaurentSeries R
        h : Not (LE.le 0 (HahnSeries.order z))
        ⊢ Eq ((HahnSeries.ofPowerSeries Int R) (HPow.hPow PowerSeries.X (HahnSeries.or …
      -/
      rw [ofPowerSeries_X_pow, Int.ofNat_natAbs_of_nonpos]
      /-
        case neg
        R : Type u_1
        inst✝ : CommRing R
        z : LaurentSeries R
        h : Not (LE.le 0 (HahnSeries.order z))
        ⊢ LE.le (HahnSeries.order z) 0
      -/
      exact le_of_not_ge h
      /-
        🎉 no goals
      -/
  exists_of_eq {x y} := by
    /-
      R : Type u_1
      inst✝ : CommRing R
      x y : PowerSeries R
      ⊢ Eq ((algebraMap (PowerSeries R) (LaurentSeries R)) x) ((algebraMap (PowerSer …
    -/
    rw [coe_algebraMap, ofPowerSeries_injective.eq_iff]
    /-
      R : Type u_1
      inst✝ : CommRing R
      x y : PowerSeries R
      ⊢ Eq x y → Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
    -/
    rintro rfl
    /-
      R : Type u_1
      inst✝ : CommRing R
      x : PowerSeries R
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) x)
    -/
    exact ⟨1, rfl⟩
    /-
      🎉 no goals
    -/


instance {K : Type*} [Field K] : IsFractionRing K⟦X⟧ K⸨X⸩ :=
  IsLocalization.of_le (Submonoid.powers (PowerSeries.X : K⟦X⟧)) _
    (powers_le_nonZeroDivisors_of_noZeroDivisors PowerSeries.X_ne_zero) fun _ hf =>
    isUnit_of_mem_nonZeroDivisors <| map_mem_nonZeroDivisors _ HahnSeries.ofPowerSeries_injective hf


@[norm_cast]
theorem coe_zero : ((0 : R⟦X⟧) : R⸨X⸩) = 0 :=
  (ofPowerSeries ℤ R).map_zero


@[norm_cast]
theorem coe_one : ((1 : R⟦X⟧) : R⸨X⸩) = 1 :=
  (ofPowerSeries ℤ R).map_one


@[norm_cast]
theorem coe_add : ((f + g : R⟦X⟧) : R⸨X⸩) = f + g :=
  (ofPowerSeries ℤ R).map_add _ _


@[norm_cast]
theorem coe_sub : ((f' - g' : R'⟦X⟧) : R'⸨X⸩) = f' - g' :=
  (ofPowerSeries ℤ R').map_sub _ _


@[norm_cast]
theorem coe_neg : ((-f' : R'⟦X⟧) : R'⸨X⸩) = -f' :=
  (ofPowerSeries ℤ R').map_neg _


@[norm_cast]
theorem coe_mul : ((f * g : R⟦X⟧) : R⸨X⸩) = f * g :=
  (ofPowerSeries ℤ R).map_mul _ _


theorem coeff_coe (i : ℤ) :
    ((f : R⟦X⟧) : R⸨X⸩).coeff i =
      if i < 0 then 0 else PowerSeries.coeff R i.natAbs f := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : PowerSeries R
    i : Int
    ⊢ Eq (((HahnSeries.ofPowerSeries Int R) f).coeff i) (ite (LT.lt i 0) 0 ((Power …
  -/
  cases i
  · rw [Int.ofNat_eq_coe, coeff_coe_powerSeries, if_neg (Int.natCast_nonneg _).not_lt,
      Int.natAbs_ofNat]
    /-
      case negSucc
      R : Type u_1
      inst✝ : Semiring R
      f : PowerSeries R
      a✝ : Nat
      ⊢ Eq (((HahnSeries.ofPowerSeries Int R) f).coeff (Int.negSucc a✝)) (ite (LT.lt …
    -/
  · rw [ofPowerSeries_apply, embDomain_notin_image_support, if_pos (Int.negSucc_lt_zero _)]
    simp only [not_exists, RelEmbedding.coe_mk, Set.mem_image, not_and, Function.Embedding.coeFn_mk,
      Ne, toPowerSeries_symm_apply_coeff, mem_support, imp_true_iff,
      not_false_iff, reduceCtorEq]


theorem coe_C (r : R) : ((C R r : R⟦X⟧) : R⸨X⸩) = HahnSeries.C r :=
  ofPowerSeries_C _


theorem coe_X : ((X : R⟦X⟧) : R⸨X⸩) = single 1 1 :=
  ofPowerSeries_X


@[simp, norm_cast]
theorem coe_smul {S : Type*} [Semiring S] [Module R S] (r : R) (x : S⟦X⟧) :
    ((r • x : S⟦X⟧) : S⸨X⸩) = r • (ofPowerSeries ℤ S x) := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    S : Type u_3
    inst✝¹ : Semiring S
    inst✝ : Module R S
    r : R
    x : PowerSeries S
    ⊢ Eq ((HahnSeries.ofPowerSeries Int S) (HSMul.hSMul r x)) (HSMul.hSMul r ((Hah …
  -/
  ext
  /-
    case coeff.h
    R : Type u_1
    inst✝² : Semiring R
    S : Type u_3
    inst✝¹ : Semiring S
    inst✝ : Module R S
    r : R
    x : PowerSeries S
    x✝ : Int
    ⊢ Eq (((HahnSeries.ofPowerSeries Int S) (HSMul.hSMul r x)).coeff x✝) ((HSMul.h …
  -/
  simp [coeff_coe, coeff_smul, smul_ite]
  /-
    🎉 no goals
  -/

-- Porting note: RingHom.map_bit0 and RingHom.map_bit1 no longer exist


@[norm_cast]
theorem coe_pow (n : ℕ) : ((f ^ n : R⟦X⟧) : R⸨X⸩) = (ofPowerSeries ℤ R f) ^ n :=
  (ofPowerSeries ℤ R).map_pow _ _


/-- The coercion `RatFunc F → F⸨X⸩` as bundled alg hom. -/
def coeAlgHom (F : Type u) [Field F] : RatFunc F →ₐ[F[X]] F⸨X⸩ :=
  liftAlgHom (Algebra.ofId _ _) <|
    nonZeroDivisors_le_comap_nonZeroDivisors_of_injective _ <|
      Polynomial.algebraMap_hahnSeries_injective _


/-- The coercion `RatFunc F → F⸨X⸩` as a function.

This is the implementation of `coeToLaurentSeries`.
-/
@[coe]
def coeToLaurentSeries_fun {F : Type u} [Field F] : RatFunc F → F⸨X⸩ :=
  coeAlgHom F


instance coeToLaurentSeries : Coe (RatFunc F) F⸨X⸩ :=
  ⟨coeToLaurentSeries_fun⟩


theorem coe_def : (f : F⸨X⸩) = coeAlgHom F f :=
  rfl


attribute [-instance] RatFunc.instCoePolynomial in
-- avoids a diamond, see https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/compiling.20behaviour.20within.20one.20file
theorem coe_num_denom : (f : F⸨X⸩) = f.num / f.denom :=
  liftAlgHom_apply _ _ f


theorem coe_injective : Function.Injective ((↑) : RatFunc F → F⸨X⸩) :=
  liftAlgHom_injective _ (Polynomial.algebraMap_hahnSeries_injective _)

-- Porting note: removed the `norm_cast` tag:
-- `norm_cast: badly shaped lemma, rhs can't start with coe `↑(coeAlgHom F) f`

@[simp]
theorem coe_apply : coeAlgHom F f = f :=
  rfl

-- avoids a diamond, see https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/compiling.20behaviour.20within.20one.20file

theorem coe_coe (P : Polynomial F) : ((P : F⟦X⟧) : F⸨X⸩) = (P : RatFunc F) := by
  /-
    F : Type u
    inst✝ : Field F
    P : Polynomial F
    ⊢ Eq ((HahnSeries.ofPowerSeries Int F) ↑P) ↑↑P
  -/
  simp only [coePolynomial, coe_def, AlgHom.commutes, algebraMap_hahnSeries_apply]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_zero : ((0 : RatFunc F) : F⸨X⸩) = 0 :=
  map_zero (coeAlgHom F)


theorem coe_ne_zero {f : Polynomial F} (hf : f ≠ 0) : (↑f : F⟦X⟧) ≠ 0 := by
  /-
    F : Type u
    inst✝ : Field F
    f : Polynomial F
    hf : Ne f 0
    ⊢ Ne (↑f) 0
  -/
  simp only [ne_eq, Polynomial.coe_eq_zero_iff, hf, not_false_eq_true]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_one : ((1 : RatFunc F) : F⸨X⸩) = 1 :=
  map_one (coeAlgHom F)


@[simp, norm_cast]
theorem coe_add : ((f + g : RatFunc F) : F⸨X⸩) = f + g :=
  map_add (coeAlgHom F) _ _


@[simp, norm_cast]
theorem coe_sub : ((f - g : RatFunc F) : F⸨X⸩) = f - g :=
  map_sub (coeAlgHom F) _ _


@[simp, norm_cast]
theorem coe_neg : ((-f : RatFunc F) : F⸨X⸩) = -f :=
  map_neg (coeAlgHom F) _


@[simp, norm_cast]
theorem coe_mul : ((f * g : RatFunc F) : F⸨X⸩) = f * g :=
  map_mul (coeAlgHom F) _ _


@[simp, norm_cast]
theorem coe_pow (n : ℕ) : ((f ^ n : RatFunc F) : F⸨X⸩) = (f : F⸨X⸩) ^ n :=
  map_pow (coeAlgHom F) _ _


@[simp, norm_cast]
theorem coe_div : ((f / g : RatFunc F) : F⸨X⸩) = (f : F⸨X⸩) / (g : F⸨X⸩) :=
  map_div₀ (coeAlgHom F) _ _


@[simp, norm_cast]
theorem coe_C (r : F) : ((RatFunc.C r : RatFunc F) : F⸨X⸩) = HahnSeries.C r := by
  rw [coe_num_denom, num_C, denom_C, Polynomial.coe_C, -- Porting note: removed `coe_C`
    Polynomial.coe_one,
    PowerSeries.coe_one, div_one]
  /-
    F : Type u
    inst✝ : Field F
    r : F
    ⊢ Eq ((HahnSeries.ofPowerSeries Int F) ((PowerSeries.C F) r)) (HahnSeries.C r)
  -/
  simp only [algebraMap_eq_C, ofPowerSeries_C, C_apply]  -- Porting note: added
  /-
    🎉 no goals
  -/

-- TODO: generalize over other modules

@[simp, norm_cast]
theorem coe_smul (r : F) : ((r • f : RatFunc F) : F⸨X⸩) = r • (f : F⸨X⸩) := by
  /-
    F : Type u
    inst✝ : Field F
    f : RatFunc F
    r : F
    ⊢ Eq (↑(HSMul.hSMul r f)) (HSMul.hSMul r ↑f)
  -/
  rw [RatFunc.smul_eq_C_mul, ← C_mul_eq_smul, coe_mul, coe_C]
  /-
    🎉 no goals
  -/

-- Porting note: removed `norm_cast` because "badly shaped lemma, rhs can't start with coe"
-- even though `single 1 1` is a bundled function application, not a "real" coercion

@[simp]
theorem coe_X : ((X : RatFunc F) : F⸨X⸩) = single 1 1 := by
  rw [coe_num_denom, num_X, denom_X, Polynomial.coe_X, -- Porting note: removed `coe_C`
     Polynomial.coe_one,
     PowerSeries.coe_one, div_one]
  /-
    F : Type u
    inst✝ : Field F
    ⊢ Eq ((HahnSeries.ofPowerSeries Int F) PowerSeries.X) ((HahnSeries.single 1) 1)
  -/
  simp only [ofPowerSeries_X]  -- Porting note: added
  /-
    🎉 no goals
  -/


theorem single_one_eq_pow {R : Type _} [Ring R] (n : ℕ) :
    single (n : ℤ) (1 : R) = single (1 : ℤ) 1 ^ n := by
  /-
    R : Type u_2
    inst✝ : Ring R
    n : Nat
    ⊢ Eq ((HahnSeries.single ↑n) 1) (HPow.hPow ((HahnSeries.single 1) 1) n)
  -/
  induction' n with n h_ind
    /-
      case zero
      R : Type u_2
      inst✝ : Ring R
      ⊢ Eq ((HahnSeries.single ↑0) 1) (HPow.hPow ((HahnSeries.single 1) 1) 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [← Int.ofNat_add_one_out, pow_succ', ← h_ind, HahnSeries.single_mul_single, one_mul,
      add_comm]


theorem single_inv (d : ℤ) {α : F} (hα : α ≠ 0) :
    single (-d) (α⁻¹ : F) = (single (d : ℤ) (α : F))⁻¹ := by
  /-
    F : Type u
    inst✝ : Field F
    d : Int
    α : F
    hα : Ne α 0
    ⊢ Eq ((HahnSeries.single (Neg.neg d)) (Inv.inv α)) (Inv.inv ((HahnSeries.singl …
  -/
  apply eq_inv_of_mul_eq_one_right
  /-
    case h
    F : Type u
    inst✝ : Field F
    d : Int
    α : F
    hα : Ne α 0
    ⊢ Eq (HMul.hMul ((HahnSeries.single d) α) ((HahnSeries.single (Neg.neg d)) (In …
  -/
  simp [hα]
  /-
    🎉 no goals
  -/


theorem single_zpow (n : ℤ) :
    single (n : ℤ) (1 : F) = single (1 : ℤ) 1 ^ n := by
  /-
    F : Type u
    inst✝ : Field F
    n : Int
    ⊢ Eq ((HahnSeries.single n) 1) (HPow.hPow ((HahnSeries.single 1) 1) n)
  -/
  induction' n with n_pos n_neg
    /-
      case ofNat
      F : Type u
      inst✝ : Field F
      n_pos : Nat
      ⊢ Eq ((HahnSeries.single (Int.ofNat n_pos)) 1) (HPow.hPow ((HahnSeries.single  …
    -/
  · apply single_one_eq_pow
    /-
      🎉 no goals
    -/
  · rw [Int.negSucc_coe, Int.ofNat_add, Nat.cast_one, ← inv_one,
      single_inv (n_neg + 1 : ℤ) one_ne_zero, zpow_neg, ← Nat.cast_one, ← Int.ofNat_add,
      Nat.cast_one, inv_inj, zpow_natCast, single_one_eq_pow, inv_one]


instance : Algebra (RatFunc F) F⸨X⸩ := RingHom.toAlgebra (coeAlgHom F).toRingHom


theorem algebraMap_apply_div :
    algebraMap (RatFunc F) F⸨X⸩ (algebraMap _ _ p / algebraMap _ _ q) =
      algebraMap F[X] F⸨X⸩ p / algebraMap _ _ q := by
  -- Porting note: had to supply implicit arguments to `convert`
  /-
    F : Type u
    inst✝ : Field F
    p q : Polynomial F
    ⊢ Eq ((algebraMap (RatFunc F) (LaurentSeries F)) (HDiv.hDiv ((algebraMap (Poly …
  -/
  convert coe_div (algebraMap F[X] (RatFunc F) p) (algebraMap F[X] (RatFunc F) q) <;>
    rw [← mk_one, coe_def, coeAlgHom, mk_eq_div, liftAlgHom_apply_div, map_one, div_one,
      Algebra.ofId_apply]


instance : IsScalarTower F[X] (RatFunc F) F⸨X⸩ :=
  ⟨fun x y z => by
    /-
      R : Type u_1
      F : Type u
      inst✝ : Field F
      p q : Polynomial F
      f g : RatFunc F
      x : Polynomial F
      y : RatFunc F
      z : LaurentSeries F
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul x y) z) (HSMul.hSMul x (HSMul.hSMul y z))
    -/
    ext
    /-
      case coeff.h
      R : Type u_1
      F : Type u
      inst✝ : Field F
      p q : Polynomial F
      f g : RatFunc F
      x : Polynomial F
      y : RatFunc F
      z : LaurentSeries F
      x✝ : Int
      ⊢ Eq ((HSMul.hSMul (HSMul.hSMul x y) z).coeff x✝) ((HSMul.hSMul x (HSMul.hSMul …
    -/
    simp⟩
    /-
      🎉 no goals
    -/


/-- The prime ideal `(X)` of `K⟦X⟧`, when `K` is a field, as a term of the `HeightOneSpectrum`. -/
def idealX : IsDedekindDomain.HeightOneSpectrum K⟦X⟧ where
  asIdeal := Ideal.span {X}
  isPrime := PowerSeries.span_X_isPrime
                /-
                  R : Type u_1
                  K : Type u_2
                  inst✝ : Field K
                  ⊢ Ne (Ideal.span (Singleton.singleton PowerSeries.X)) Bot.bot
                -/
  ne_bot  := by rw [ne_eq, Ideal.span_singleton_eq_bot]; exact X_ne_zero
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem intValuation_eq_of_coe (P : K[X]) :
    (Polynomial.idealX K).intValuation P = (idealX K).intValuation (P : K⟦X⟧) := by
  /-
    K : Type u_2
    inst✝ : Field K
    P : Polynomial K
    ⊢ Eq ((Polynomial.idealX K).intValuation P) ((PowerSeries.idealX K).intValuati …
  -/
  by_cases hP : P = 0
    /-
      case pos
      K : Type u_2
      inst✝ : Field K
      P : Polynomial K
      hP : Eq P 0
      ⊢ Eq ((Polynomial.idealX K).intValuation P) ((PowerSeries.idealX K).intValuati …
    -/
  · rw [hP, Valuation.map_zero, Polynomial.coe_zero, Valuation.map_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u_2
    inst✝ : Field K
    P : Polynomial K
    hP : Not (Eq P 0)
    ⊢ Eq ((Polynomial.idealX K).intValuation P) ((PowerSeries.idealX K).intValuati …
  -/
  simp only [intValuation_apply]
  /-
    case neg
    K : Type u_2
    inst✝ : Field K
    P : Polynomial K
    hP : Not (Eq P 0)
    ⊢ Eq ((Polynomial.idealX K).intValuationDef P) ((PowerSeries.idealX K).intValu …
  -/
  rw [intValuationDef_if_neg _ hP, intValuationDef_if_neg _ <| coe_ne_zero hP]
  simp only [idealX_span, ofAdd_neg, inv_inj, WithZero.coe_inj, EmbeddingLike.apply_eq_iff_eq,
    Nat.cast_inj]
  have span_ne_zero :
    (Ideal.span {P} : Ideal K[X]) ≠ 0 ∧ (Ideal.span {Polynomial.X} : Ideal K[X]) ≠ 0 := by
    simp only [Ideal.zero_eq_bot, ne_eq, Ideal.span_singleton_eq_bot, hP, Polynomial.X_ne_zero,
      not_false_iff, and_self_iff]
  have span_ne_zero' :
    (Ideal.span {↑P} : Ideal K⟦X⟧) ≠ 0 ∧ ((idealX K).asIdeal : Ideal K⟦X⟧) ≠ 0 := by
    simp only [Ideal.zero_eq_bot, ne_eq, Ideal.span_singleton_eq_bot, coe_eq_zero_iff, hP,
      not_false_eq_true, true_and, (idealX K).3]
  rw [count_associates_factors_eq  (span_ne_zero).1
    (Ideal.span_singleton_prime Polynomial.X_ne_zero|>.mpr prime_X) (span_ne_zero).2,
    count_associates_factors_eq]
  /-
    case neg
    K : Type u_2
    inst✝ : Field K
    P : Polynomial K
    hP : Not (Eq P 0)
    span_ne_zero : And (Ne (Ideal.span (Singleton.singleton P)) 0) (Ne (Ideal.span …
    span_ne_zero' : And (Ne (Ideal.span (Singleton.singleton ↑P)) 0) (Ne (PowerSer …
    ⊢ Eq (Multiset.count (Ideal.span (Singleton.singleton Polynomial.X)) (UniqueFa …
  -/
  on_goal 1 => convert (normalized_count_X_eq_of_coe hP).symm
  exacts [count_span_normalizedFactors_eq_of_normUnit hP Polynomial.normUnit_X prime_X,
    count_span_normalizedFactors_eq_of_normUnit (coe_ne_zero hP) normUnit_X X_prime,
    span_ne_zero'.1, (idealX K).isPrime, span_ne_zero'.2]


/-- The integral valuation of the power series `X : K⟦X⟧` equals `(ofAdd -1) : ℤₘ₀`-/
@[simp]
theorem intValuation_X : (idealX K).intValuationDef X = ↑(Multiplicative.ofAdd (-1 : ℤ)) := by
  /-
    K : Type u_2
    inst✝ : Field K
    ⊢ Eq ((PowerSeries.idealX K).intValuationDef PowerSeries.X) ↑(Multiplicative.o …
  -/
  rw [← Polynomial.coe_X, ← intValuation_apply, ← intValuation_eq_of_coe]
  /-
    K : Type u_2
    inst✝ : Field K
    ⊢ Eq ((Polynomial.idealX K).intValuation Polynomial.X) ↑(Multiplicative.ofAdd  …
  -/
  apply intValuation_singleton _ Polynomial.X_ne_zero (by rfl)
  /-
    🎉 no goals
  -/


theorem valuation_eq_LaurentSeries_valuation (P : RatFunc K) :
    (Polynomial.idealX K).valuation P = (PowerSeries.idealX K).valuation (P : K⸨X⸩) := by
  /-
    K : Type u_2
    inst✝ : Field K
    P : RatFunc K
    ⊢ Eq ((Polynomial.idealX K).valuation P) ((PowerSeries.idealX K).valuation ↑P)
  -/
  refine RatFunc.induction_on' P ?_
  /-
    K : Type u_2
    inst✝ : Field K
    P : RatFunc K
    ⊢ ∀ (p q : Polynomial K), Ne q 0 → Eq ((Polynomial.idealX K).valuation (RatFun …
  -/
  intro f g h
  /-
    K : Type u_2
    inst✝ : Field K
    P : RatFunc K
    f g : Polynomial K
    h : Ne g 0
    ⊢ Eq ((Polynomial.idealX K).valuation (RatFunc.mk f g)) ((PowerSeries.idealX K …
  -/
  rw [Polynomial.valuation_of_mk K f h, RatFunc.mk_eq_mk' f h, Eq.comm]
  convert @valuation_of_mk' K⟦X⟧ _ _ K⸨X⸩ _ _ _ (PowerSeries.idealX K) f
        ⟨g, mem_nonZeroDivisors_iff_ne_zero.2 <| coe_ne_zero h⟩
    /-
      case h.e'_2.h.e'_6
      K : Type u_2
      inst✝ : Field K
      P : RatFunc K
      f g : Polynomial K
      h : Ne g 0
      ⊢ Eq (↑(IsLocalization.mk' (RatFunc K) f ⟨g, ⋯⟩)) (IsLocalization.mk' (Laurent …
    -/
  · simp only [IsFractionRing.mk'_eq_div, coe_div, LaurentSeries.coe_algebraMap, coe_coe]
    /-
      case h.e'_2.h.e'_6
      K : Type u_2
      inst✝ : Field K
      P : RatFunc K
      f g : Polynomial K
      h : Ne g 0
      ⊢ Eq (HDiv.hDiv ↑((algebraMap (Polynomial K) (RatFunc K)) f) ↑((algebraMap (Po …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case h.e'_3.h.e'_5
    K : Type u_2
    inst✝ : Field K
    P : RatFunc K
    f g : Polynomial K
    h : Ne g 0
    ⊢ Eq ((Polynomial.idealX K).intValuation f) ((PowerSeries.idealX K).intValuati …
  -/
  exacts [intValuation_eq_of_coe _, intValuation_eq_of_coe _]
  /-
    🎉 no goals
  -/


instance : Valued K⸨X⸩ ℤₘ₀ := Valued.mk' (PowerSeries.idealX K).valuation


theorem valuation_X_pow (s : ℕ) :
    Valued.v (((X : K⟦X⟧) : K⸨X⸩) ^ s) = Multiplicative.ofAdd (-(s : ℤ)) := by
  erw [map_pow, ← one_mul (s : ℤ), ← neg_mul (1 : ℤ) s, Int.ofAdd_mul,
    WithZero.coe_zpow, ofAdd_neg, WithZero.coe_inv, zpow_natCast, valuation_of_algebraMap,
    intValuation_toFun, intValuation_X, ofAdd_neg, WithZero.coe_inv, inv_pow]


theorem valuation_single_zpow (s : ℤ) :
    Valued.v (HahnSeries.single s (1 : K) : K⸨X⸩) =
      Multiplicative.ofAdd (-(s : ℤ)) := by
  /-
    K : Type u_2
    inst✝ : Field K
    s : Int
    ⊢ Eq (Valued.v ((HahnSeries.single s) 1)) ↑(Multiplicative.ofAdd (Neg.neg s))
  -/
  have : Valued.v (1 : K⸨X⸩) = (1 : ℤₘ₀) := Valued.v.map_one
  rw [← single_zero_one, ← add_neg_cancel s, ← mul_one 1, ← single_mul_single, map_mul,
    mul_eq_one_iff_eq_inv₀] at this
    /-
      K : Type u_2
      inst✝ : Field K
      s : Int
      this : Eq (Valued.v ((HahnSeries.single s) 1)) (Inv.inv (Valued.v ((HahnSeries …
      ⊢ Eq (Valued.v ((HahnSeries.single s) 1)) ↑(Multiplicative.ofAdd (Neg.neg s))
    -/
  · rw [this]
    /-
      K : Type u_2
      inst✝ : Field K
      s : Int
      this : Eq (Valued.v ((HahnSeries.single s) 1)) (Inv.inv (Valued.v ((HahnSeries …
      ⊢ Eq (Inv.inv (Valued.v ((HahnSeries.single (Neg.neg s)) 1))) ↑(Multiplicative …
    -/
    induction' s with s s
      /-
        case ofNat
        K : Type u_2
        inst✝ : Field K
        s : Nat
        this : Eq (Valued.v ((HahnSeries.single (Int.ofNat s)) 1)) (Inv.inv (Valued.v  …
        ⊢ Eq (Inv.inv (Valued.v ((HahnSeries.single (Neg.neg (Int.ofNat s))) 1))) ↑(Mu …
      -/
    · rw [Int.ofNat_eq_coe, ← HahnSeries.ofPowerSeries_X_pow] at this
      /-
        case ofNat
        K : Type u_2
        inst✝ : Field K
        s : Nat
        this : Eq (Valued.v ((HahnSeries.ofPowerSeries Int K) (HPow.hPow PowerSeries.X …
        ⊢ Eq (Inv.inv (Valued.v ((HahnSeries.single (Neg.neg (Int.ofNat s))) 1))) ↑(Mu …
      -/
      rw [Int.ofNat_eq_coe, ← this, PowerSeries.coe_pow, valuation_X_pow]
      /-
        🎉 no goals
      -/
    · simp only [Int.negSucc_coe, neg_neg, ← HahnSeries.ofPowerSeries_X_pow, PowerSeries.coe_pow,
        valuation_X_pow, ofAdd_neg, WithZero.coe_inv, inv_inv]
    /-
      K : Type u_2
      inst✝ : Field K
      s : Int
      this : Eq (HMul.hMul (Valued.v ((HahnSeries.single s) 1)) (Valued.v ((HahnSeri …
      ⊢ Ne (Valued.v ((HahnSeries.single (Neg.neg s)) 1)) 0
    -/
  · simp only [Valuation.ne_zero_iff, ne_eq, one_ne_zero, not_false_iff, HahnSeries.single_ne_zero]
    /-
      🎉 no goals
    -/

/- The coefficients of a power series vanish in degree strictly less than its valuation. -/

theorem coeff_zero_of_lt_intValuation {n d : ℕ} {f : K⟦X⟧}
    (H : Valued.v (f : K⸨X⸩) ≤ Multiplicative.ofAdd (-d : ℤ)) :
    n < d → coeff K n f = 0 := by
  /-
    K : Type u_2
    inst✝ : Field K
    n d : Nat
    f : PowerSeries K
    H : LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f)) ↑(Multiplicative.ofA …
    ⊢ LT.lt n d → Eq ((PowerSeries.coeff K n) f) 0
  -/
  intro hnd
  /-
    K : Type u_2
    inst✝ : Field K
    n d : Nat
    f : PowerSeries K
    H : LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f)) ↑(Multiplicative.ofA …
    hnd : LT.lt n d
    ⊢ Eq ((PowerSeries.coeff K n) f) 0
  -/
  apply (PowerSeries.X_pow_dvd_iff).mp _ n hnd
  erw [← span_singleton_dvd_span_singleton_iff_dvd, ← Ideal.span_singleton_pow,
    ← (intValuation_le_pow_iff_dvd (PowerSeries.idealX K) f d), ← intValuation_apply,
    ← valuation_of_algebraMap (R := K⟦X⟧) (K := K⸨X⸩)]
  /-
    K : Type u_2
    inst✝ : Field K
    n d : Nat
    f : PowerSeries K
    H : LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f)) ↑(Multiplicative.ofA …
    hnd : LT.lt n d
    ⊢ LE.le ((PowerSeries.idealX K).valuation ((algebraMap (PowerSeries K) (Lauren …
  -/
  exact H
  /-
    🎉 no goals
  -/

/- The valuation of a power series is the order of the first non-zero coefficient. -/

theorem intValuation_le_iff_coeff_lt_eq_zero {d : ℕ} (f : K⟦X⟧) :
    Valued.v (f : K⸨X⸩) ≤ Multiplicative.ofAdd (-d : ℤ) ↔
      ∀ n : ℕ, n < d → coeff K n f = 0 := by
  have : PowerSeries.X ^ d ∣ f ↔ ∀ n : ℕ, n < d → (PowerSeries.coeff K n) f = 0 :=
    ⟨PowerSeries.X_pow_dvd_iff.mp, PowerSeries.X_pow_dvd_iff.mpr⟩
  erw [← this, valuation_of_algebraMap (PowerSeries.idealX K) f, ←
    span_singleton_dvd_span_singleton_iff_dvd, ← Ideal.span_singleton_pow]
  /-
    K : Type u_2
    inst✝ : Field K
    d : Nat
    f : PowerSeries K
    this : Iff (Dvd.dvd (HPow.hPow PowerSeries.X d) f) (∀ (n : Nat), LT.lt n d → E …
    ⊢ Iff (LE.le ((PowerSeries.idealX K).intValuation f) ↑(Multiplicative.ofAdd (N …
  -/
  apply intValuation_le_pow_iff_dvd
  /-
    🎉 no goals
  -/

/- The coefficients of a Laurent series vanish in degree strictly less than its valuation. -/

theorem coeff_zero_of_lt_valuation {n D : ℤ} {f : K⸨X⸩}
    (H : Valued.v f ≤ Multiplicative.ofAdd (-D)) : n < D → f.coeff n = 0 := by
  /-
    K : Type u_2
    inst✝ : Field K
    n D : Int
    f : LaurentSeries K
    H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
    ⊢ LT.lt n D → Eq (f.coeff n) 0
  -/
  intro hnd
  /-
    K : Type u_2
    inst✝ : Field K
    n D : Int
    f : LaurentSeries K
    H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
    hnd : LT.lt n D
    ⊢ Eq (f.coeff n) 0
  -/
  by_cases h_n_ord : n < f.order
    /-
      case pos
      K : Type u_2
      inst✝ : Field K
      n D : Int
      f : LaurentSeries K
      H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
      hnd : LT.lt n D
      h_n_ord : LT.lt n (HahnSeries.order f)
      ⊢ Eq (f.coeff n) 0
    -/
  · exact coeff_eq_zero_of_lt_order h_n_ord
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u_2
    inst✝ : Field K
    n D : Int
    f : LaurentSeries K
    H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
    hnd : LT.lt n D
    h_n_ord : Not (LT.lt n (HahnSeries.order f))
    ⊢ Eq (f.coeff n) 0
  -/
  rw [not_lt] at h_n_ord
  /-
    case neg
    K : Type u_2
    inst✝ : Field K
    n D : Int
    f : LaurentSeries K
    H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
    hnd : LT.lt n D
    h_n_ord : LE.le (HahnSeries.order f) n
    ⊢ Eq (f.coeff n) 0
  -/
  set F := powerSeriesPart f with hF
  /-
    case neg
    K : Type u_2
    inst✝ : Field K
    n D : Int
    f : LaurentSeries K
    H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
    hnd : LT.lt n D
    h_n_ord : LE.le (HahnSeries.order f) n
    F : PowerSeries K := f.powerSeriesPart
    hF : Eq F f.powerSeriesPart
    ⊢ Eq (f.coeff n) 0
  -/
  by_cases ord_nonpos : f.order ≤ 0
    /-
      case pos
      K : Type u_2
      inst✝ : Field K
      n D : Int
      f : LaurentSeries K
      H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
      hnd : LT.lt n D
      h_n_ord : LE.le (HahnSeries.order f) n
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LE.le (HahnSeries.order f) 0
      ⊢ Eq (f.coeff n) 0
    -/
  · obtain ⟨s, hs⟩ := Int.exists_eq_neg_ofNat ord_nonpos
    /-
      case pos.intro
      K : Type u_2
      inst✝ : Field K
      n D : Int
      f : LaurentSeries K
      H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
      hnd : LT.lt n D
      h_n_ord : LE.le (HahnSeries.order f) n
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LE.le (HahnSeries.order f) 0
      s : Nat
      hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
      ⊢ Eq (f.coeff n) 0
    -/
    obtain ⟨m, hm⟩ := Int.eq_ofNat_of_zero_le (neg_le_iff_add_nonneg.mp (hs ▸ h_n_ord))
    /-
      case pos.intro.intro
      K : Type u_2
      inst✝ : Field K
      n D : Int
      f : LaurentSeries K
      H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
      hnd : LT.lt n D
      h_n_ord : LE.le (HahnSeries.order f) n
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LE.le (HahnSeries.order f) 0
      s : Nat
      hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
      m : Nat
      hm : Eq (HAdd.hAdd n ↑s) ↑m
      ⊢ Eq (f.coeff n) 0
    -/
    obtain ⟨d, hd⟩ := Int.eq_ofNat_of_zero_le (a := D + s) (by omega)
    /-
      case pos.intro.intro.intro
      K : Type u_2
      inst✝ : Field K
      n D : Int
      f : LaurentSeries K
      H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
      hnd : LT.lt n D
      h_n_ord : LE.le (HahnSeries.order f) n
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LE.le (HahnSeries.order f) 0
      s : Nat
      hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
      m : Nat
      hm : Eq (HAdd.hAdd n ↑s) ↑m
      d : Nat
      hd : Eq (HAdd.hAdd D ↑s) ↑d
      ⊢ Eq (f.coeff n) 0
    -/
    rw [eq_add_neg_of_add_eq hm, add_comm, ← hs, ← powerSeriesPart_coeff]
    /-
      case pos.intro.intro.intro
      K : Type u_2
      inst✝ : Field K
      n D : Int
      f : LaurentSeries K
      H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
      hnd : LT.lt n D
      h_n_ord : LE.le (HahnSeries.order f) n
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LE.le (HahnSeries.order f) 0
      s : Nat
      hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
      m : Nat
      hm : Eq (HAdd.hAdd n ↑s) ↑m
      d : Nat
      hd : Eq (HAdd.hAdd D ↑s) ↑d
      ⊢ Eq ((PowerSeries.coeff K m) f.powerSeriesPart) 0
    -/
    apply (intValuation_le_iff_coeff_lt_eq_zero K F).mp _ m (by linarith)
    rwa [hF, ofPowerSeries_powerSeriesPart f, hs, neg_neg, ← hd, neg_add_rev, ofAdd_add, map_mul,
      ← ofPowerSeries_X_pow s, PowerSeries.coe_pow,  WithZero.coe_mul, valuation_X_pow K s,
      mul_le_mul_left (by simp only [ne_eq, WithZero.coe_ne_zero, not_false_iff, zero_lt_iff])]
    /-
      case neg
      K : Type u_2
      inst✝ : Field K
      n D : Int
      f : LaurentSeries K
      H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
      hnd : LT.lt n D
      h_n_ord : LE.le (HahnSeries.order f) n
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
      ⊢ Eq (f.coeff n) 0
    -/
  · rw [not_le] at ord_nonpos
    /-
      case neg
      K : Type u_2
      inst✝ : Field K
      n D : Int
      f : LaurentSeries K
      H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
      hnd : LT.lt n D
      h_n_ord : LE.le (HahnSeries.order f) n
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LT.lt 0 (HahnSeries.order f)
      ⊢ Eq (f.coeff n) 0
    -/
    obtain ⟨s, hs⟩ := Int.exists_eq_neg_ofNat (Int.neg_nonpos_of_nonneg (le_of_lt ord_nonpos))
    /-
      case neg.intro
      K : Type u_2
      inst✝ : Field K
      n D : Int
      f : LaurentSeries K
      H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
      hnd : LT.lt n D
      h_n_ord : LE.le (HahnSeries.order f) n
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LT.lt 0 (HahnSeries.order f)
      s : Nat
      hs : Eq (Neg.neg (HahnSeries.order f)) (Neg.neg ↑s)
      ⊢ Eq (f.coeff n) 0
    -/
    obtain ⟨m, hm⟩ := Int.eq_ofNat_of_zero_le (a := n - s) (by omega)
    /-
      case neg.intro.intro
      K : Type u_2
      inst✝ : Field K
      n D : Int
      f : LaurentSeries K
      H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
      hnd : LT.lt n D
      h_n_ord : LE.le (HahnSeries.order f) n
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LT.lt 0 (HahnSeries.order f)
      s : Nat
      hs : Eq (Neg.neg (HahnSeries.order f)) (Neg.neg ↑s)
      m : Nat
      hm : Eq (HSub.hSub n ↑s) ↑m
      ⊢ Eq (f.coeff n) 0
    -/
    obtain ⟨d, hd⟩ := Int.eq_ofNat_of_zero_le (a := D - s) (by omega)
    rw [(sub_eq_iff_eq_add).mp hm, add_comm, ← neg_neg (s : ℤ), ← hs, neg_neg,
      ← powerSeriesPart_coeff]
    /-
      case neg.intro.intro.intro
      K : Type u_2
      inst✝ : Field K
      n D : Int
      f : LaurentSeries K
      H : LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
      hnd : LT.lt n D
      h_n_ord : LE.le (HahnSeries.order f) n
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LT.lt 0 (HahnSeries.order f)
      s : Nat
      hs : Eq (Neg.neg (HahnSeries.order f)) (Neg.neg ↑s)
      m : Nat
      hm : Eq (HSub.hSub n ↑s) ↑m
      d : Nat
      hd : Eq (HSub.hSub D ↑s) ↑d
      ⊢ Eq ((PowerSeries.coeff K m) f.powerSeriesPart) 0
    -/
    apply (intValuation_le_iff_coeff_lt_eq_zero K F).mp _ m (by linarith)
    rwa [hF, ofPowerSeries_powerSeriesPart f, map_mul, ← hd, hs, neg_sub, sub_eq_add_neg,
      ofAdd_add, valuation_single_zpow, neg_neg, WithZero.coe_mul,
      mul_le_mul_left (by simp only [ne_eq, WithZero.coe_ne_zero, not_false_iff, zero_lt_iff])]

/- The valuation of a Laurent series is the order of the first non-zero coefficient. -/

theorem valuation_le_iff_coeff_lt_eq_zero {D : ℤ} {f : K⸨X⸩} :
    Valued.v f ≤ ↑(Multiplicative.ofAdd (-D : ℤ)) ↔ ∀ n : ℤ, n < D → f.coeff n = 0 := by
  /-
    K : Type u_2
    inst✝ : Field K
    D : Int
    f : LaurentSeries K
    ⊢ Iff (LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))) (∀ (n : Int), L …
  -/
  refine ⟨fun hnD n hn => coeff_zero_of_lt_valuation K hnD hn, fun h_val_f => ?_⟩
  /-
    K : Type u_2
    inst✝ : Field K
    D : Int
    f : LaurentSeries K
    h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
    ⊢ LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
  -/
  let F := powerSeriesPart f
  /-
    K : Type u_2
    inst✝ : Field K
    D : Int
    f : LaurentSeries K
    h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
    F : PowerSeries K := f.powerSeriesPart
    ⊢ LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
  -/
  by_cases ord_nonpos : f.order ≤ 0
    /-
      case pos
      K : Type u_2
      inst✝ : Field K
      D : Int
      f : LaurentSeries K
      h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
      F : PowerSeries K := f.powerSeriesPart
      ord_nonpos : LE.le (HahnSeries.order f) 0
      ⊢ LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
    -/
  · obtain ⟨s, hs⟩ := Int.exists_eq_neg_ofNat ord_nonpos
    rw [← f.single_order_mul_powerSeriesPart, hs, map_mul, valuation_single_zpow, neg_neg, mul_comm,
      ← le_mul_inv_iff₀, ofAdd_neg, WithZero.coe_inv, ← mul_inv, ← WithZero.coe_mul, ← ofAdd_add,
      ← WithZero.coe_inv, ← ofAdd_neg]
      /-
        case pos.intro
        K : Type u_2
        inst✝ : Field K
        D : Int
        f : LaurentSeries K
        h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
        F : PowerSeries K := f.powerSeriesPart
        ord_nonpos : LE.le (HahnSeries.order f) 0
        s : Nat
        hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
        ⊢ LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f.powerSeriesPart)) ↑(Mult …
      -/
    · by_cases hDs : D + s ≤ 0
        /-
          case pos
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : LE.le (HahnSeries.order f) 0
          s : Nat
          hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
          hDs : LE.le (HAdd.hAdd D ↑s) 0
          ⊢ LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f.powerSeriesPart)) ↑(Mult …
        -/
      · apply le_trans ((PowerSeries.idealX K).valuation_le_one F)
        rwa [← WithZero.coe_one, ← ofAdd_zero, WithZero.coe_le_coe, Multiplicative.ofAdd_le,
          Left.nonneg_neg_iff]
        /-
          case neg
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : LE.le (HahnSeries.order f) 0
          s : Nat
          hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
          hDs : Not (LE.le (HAdd.hAdd D ↑s) 0)
          ⊢ LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f.powerSeriesPart)) ↑(Mult …
        -/
      · obtain ⟨d, hd⟩ := Int.eq_ofNat_of_zero_le (le_of_lt <| not_le.mp hDs)
        /-
          case neg.intro
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : LE.le (HahnSeries.order f) 0
          s : Nat
          hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
          hDs : Not (LE.le (HAdd.hAdd D ↑s) 0)
          d : Nat
          hd : Eq (HAdd.hAdd D ↑s) ↑d
          ⊢ LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f.powerSeriesPart)) ↑(Mult …
        -/
        rw [hd]
        /-
          case neg.intro
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : LE.le (HahnSeries.order f) 0
          s : Nat
          hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
          hDs : Not (LE.le (HAdd.hAdd D ↑s) 0)
          d : Nat
          hd : Eq (HAdd.hAdd D ↑s) ↑d
          ⊢ LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f.powerSeriesPart)) ↑(Mult …
        -/
        apply (intValuation_le_iff_coeff_lt_eq_zero K F).mpr
        /-
          case neg.intro
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : LE.le (HahnSeries.order f) 0
          s : Nat
          hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
          hDs : Not (LE.le (HAdd.hAdd D ↑s) 0)
          d : Nat
          hd : Eq (HAdd.hAdd D ↑s) ↑d
          ⊢ ∀ (n : Nat), LT.lt n d → Eq ((PowerSeries.coeff K n) F) 0
        -/
        intro n hn
        /-
          case neg.intro
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : LE.le (HahnSeries.order f) 0
          s : Nat
          hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
          hDs : Not (LE.le (HAdd.hAdd D ↑s) 0)
          d : Nat
          hd : Eq (HAdd.hAdd D ↑s) ↑d
          n : Nat
          hn : LT.lt n d
          ⊢ Eq ((PowerSeries.coeff K n) F) 0
        -/
        rw [powerSeriesPart_coeff f n, hs]
        /-
          case neg.intro
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : LE.le (HahnSeries.order f) 0
          s : Nat
          hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
          hDs : Not (LE.le (HAdd.hAdd D ↑s) 0)
          d : Nat
          hd : Eq (HAdd.hAdd D ↑s) ↑d
          n : Nat
          hn : LT.lt n d
          ⊢ Eq (f.coeff (HAdd.hAdd (Neg.neg ↑s) ↑n)) 0
        -/
        apply h_val_f
        /-
          case neg.intro.a
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : LE.le (HahnSeries.order f) 0
          s : Nat
          hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
          hDs : Not (LE.le (HAdd.hAdd D ↑s) 0)
          d : Nat
          hd : Eq (HAdd.hAdd D ↑s) ↑d
          n : Nat
          hn : LT.lt n d
          ⊢ LT.lt (HAdd.hAdd (Neg.neg ↑s) ↑n) D
        -/
        omega
        /-
          🎉 no goals
        -/
      /-
        case pos.intro
        K : Type u_2
        inst✝ : Field K
        D : Int
        f : LaurentSeries K
        h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
        F : PowerSeries K := f.powerSeriesPart
        ord_nonpos : LE.le (HahnSeries.order f) 0
        s : Nat
        hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
        ⊢ LT.lt 0 ↑(Multiplicative.ofAdd ↑s)
      -/
    · simp only [ne_eq, WithZero.coe_ne_zero, not_false_iff, zero_lt_iff]
      /-
        🎉 no goals
      -/
  · obtain ⟨s, hs⟩ := Int.exists_eq_neg_ofNat
      <| neg_nonpos_of_nonneg <| le_of_lt <| not_le.mp ord_nonpos
    /-
      case neg.intro
      K : Type u_2
      inst✝ : Field K
      D : Int
      f : LaurentSeries K
      h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
      F : PowerSeries K := f.powerSeriesPart
      ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
      s : Nat
      hs : Eq (Neg.neg (HahnSeries.order f)) (Neg.neg ↑s)
      ⊢ LE.le (Valued.v f) ↑(Multiplicative.ofAdd (Neg.neg D))
    -/
    rw [neg_inj] at hs
    rw [← f.single_order_mul_powerSeriesPart, hs, map_mul, valuation_single_zpow, mul_comm,
      ← le_mul_inv_iff₀, ofAdd_neg, WithZero.coe_inv, ← mul_inv, ← WithZero.coe_mul, ← ofAdd_add,
      ← WithZero.coe_inv, ← ofAdd_neg, neg_add, neg_neg]
      /-
        case neg.intro
        K : Type u_2
        inst✝ : Field K
        D : Int
        f : LaurentSeries K
        h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
        F : PowerSeries K := f.powerSeriesPart
        ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
        s : Nat
        hs : Eq (HahnSeries.order f) ↑s
        ⊢ LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f.powerSeriesPart)) ↑(Mult …
      -/
    · by_cases hDs : D - s ≤ 0
        /-
          case pos
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
          s : Nat
          hs : Eq (HahnSeries.order f) ↑s
          hDs : LE.le (HSub.hSub D ↑s) 0
          ⊢ LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f.powerSeriesPart)) ↑(Mult …
        -/
      · apply le_trans ((PowerSeries.idealX K).valuation_le_one F)
        /-
          case pos
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
          s : Nat
          hs : Eq (HahnSeries.order f) ↑s
          hDs : LE.le (HSub.hSub D ↑s) 0
          ⊢ LE.le 1 ↑(Multiplicative.ofAdd (HAdd.hAdd (Neg.neg D) ↑s))
        -/
        rw [← WithZero.coe_one, ← ofAdd_zero, WithZero.coe_le_coe, Multiplicative.ofAdd_le]
        /-
          case pos
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
          s : Nat
          hs : Eq (HahnSeries.order f) ↑s
          hDs : LE.le (HSub.hSub D ↑s) 0
          ⊢ LE.le 0 (HAdd.hAdd (Neg.neg D) ↑s)
        -/
        omega
        /-
          🎉 no goals
        -/
        /-
          case neg
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
          s : Nat
          hs : Eq (HahnSeries.order f) ↑s
          hDs : Not (LE.le (HSub.hSub D ↑s) 0)
          ⊢ LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f.powerSeriesPart)) ↑(Mult …
        -/
      · obtain ⟨d, hd⟩ := Int.eq_ofNat_of_zero_le (le_of_lt <| not_le.mp hDs)
        /-
          case neg.intro
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
          s : Nat
          hs : Eq (HahnSeries.order f) ↑s
          hDs : Not (LE.le (HSub.hSub D ↑s) 0)
          d : Nat
          hd : Eq (HSub.hSub D ↑s) ↑d
          ⊢ LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f.powerSeriesPart)) ↑(Mult …
        -/
        rw [← neg_neg (-D + ↑s), ← sub_eq_neg_add, neg_sub, hd]
        /-
          case neg.intro
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
          s : Nat
          hs : Eq (HahnSeries.order f) ↑s
          hDs : Not (LE.le (HSub.hSub D ↑s) 0)
          d : Nat
          hd : Eq (HSub.hSub D ↑s) ↑d
          ⊢ LE.le (Valued.v ((HahnSeries.ofPowerSeries Int K) f.powerSeriesPart)) ↑(Mult …
        -/
        apply (intValuation_le_iff_coeff_lt_eq_zero K F).mpr
        /-
          case neg.intro
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
          s : Nat
          hs : Eq (HahnSeries.order f) ↑s
          hDs : Not (LE.le (HSub.hSub D ↑s) 0)
          d : Nat
          hd : Eq (HSub.hSub D ↑s) ↑d
          ⊢ ∀ (n : Nat), LT.lt n d → Eq ((PowerSeries.coeff K n) F) 0
        -/
        intro n hn
        /-
          case neg.intro
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
          s : Nat
          hs : Eq (HahnSeries.order f) ↑s
          hDs : Not (LE.le (HSub.hSub D ↑s) 0)
          d : Nat
          hd : Eq (HSub.hSub D ↑s) ↑d
          n : Nat
          hn : LT.lt n d
          ⊢ Eq ((PowerSeries.coeff K n) F) 0
        -/
        rw [powerSeriesPart_coeff f n, hs]
        /-
          case neg.intro
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
          s : Nat
          hs : Eq (HahnSeries.order f) ↑s
          hDs : Not (LE.le (HSub.hSub D ↑s) 0)
          d : Nat
          hd : Eq (HSub.hSub D ↑s) ↑d
          n : Nat
          hn : LT.lt n d
          ⊢ Eq (f.coeff (HAdd.hAdd ↑s ↑n)) 0
        -/
        apply h_val_f (s + n)
        /-
          case neg.intro
          K : Type u_2
          inst✝ : Field K
          D : Int
          f : LaurentSeries K
          h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
          F : PowerSeries K := f.powerSeriesPart
          ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
          s : Nat
          hs : Eq (HahnSeries.order f) ↑s
          hDs : Not (LE.le (HSub.hSub D ↑s) 0)
          d : Nat
          hd : Eq (HSub.hSub D ↑s) ↑d
          n : Nat
          hn : LT.lt n d
          ⊢ LT.lt (HAdd.hAdd ↑s ↑n) D
        -/
        omega
        /-
          🎉 no goals
        -/
      /-
        case neg.intro
        K : Type u_2
        inst✝ : Field K
        D : Int
        f : LaurentSeries K
        h_val_f : ∀ (n : Int), LT.lt n D → Eq (f.coeff n) 0
        F : PowerSeries K := f.powerSeriesPart
        ord_nonpos : Not (LE.le (HahnSeries.order f) 0)
        s : Nat
        hs : Eq (HahnSeries.order f) ↑s
        ⊢ LT.lt 0 ↑(Multiplicative.ofAdd (Neg.neg ↑s))
      -/
    · simp only [ne_eq, WithZero.coe_ne_zero, not_false_iff, zero_lt_iff]
      /-
        🎉 no goals
      -/

/- Two Laurent series whose difference has small valuation have the same coefficients for
small enough indices. -/

theorem eq_coeff_of_valuation_sub_lt {d n : ℤ} {f g : K⸨X⸩}
    (H : Valued.v (g - f) ≤ ↑(Multiplicative.ofAdd (-d))) : n < d → g.coeff n = f.coeff n := by
  /-
    K : Type u_2
    inst✝ : Field K
    d n : Int
    f g : LaurentSeries K
    H : LE.le (Valued.v (HSub.hSub g f)) ↑(Multiplicative.ofAdd (Neg.neg d))
    ⊢ LT.lt n d → Eq (g.coeff n) (f.coeff n)
  -/
  by_cases triv : g = f
    /-
      case pos
      K : Type u_2
      inst✝ : Field K
      d n : Int
      f g : LaurentSeries K
      H : LE.le (Valued.v (HSub.hSub g f)) ↑(Multiplicative.ofAdd (Neg.neg d))
      triv : Eq g f
      ⊢ LT.lt n d → Eq (g.coeff n) (f.coeff n)
    -/
  · exact fun _ => by rw [triv]
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_2
      inst✝ : Field K
      d n : Int
      f g : LaurentSeries K
      H : LE.le (Valued.v (HSub.hSub g f)) ↑(Multiplicative.ofAdd (Neg.neg d))
      triv : Not (Eq g f)
      ⊢ LT.lt n d → Eq (g.coeff n) (f.coeff n)
    -/
  · intro hn
    /-
      case neg
      K : Type u_2
      inst✝ : Field K
      d n : Int
      f g : LaurentSeries K
      H : LE.le (Valued.v (HSub.hSub g f)) ↑(Multiplicative.ofAdd (Neg.neg d))
      triv : Not (Eq g f)
      hn : LT.lt n d
      ⊢ Eq (g.coeff n) (f.coeff n)
    -/
    apply eq_of_sub_eq_zero
    /-
      case neg.h
      K : Type u_2
      inst✝ : Field K
      d n : Int
      f g : LaurentSeries K
      H : LE.le (Valued.v (HSub.hSub g f)) ↑(Multiplicative.ofAdd (Neg.neg d))
      triv : Not (Eq g f)
      hn : LT.lt n d
      ⊢ Eq (HSub.hSub (g.coeff n) (f.coeff n)) 0
    -/
    rw [← HahnSeries.sub_coeff]
    /-
      case neg.h
      K : Type u_2
      inst✝ : Field K
      d n : Int
      f g : LaurentSeries K
      H : LE.le (Valued.v (HSub.hSub g f)) ↑(Multiplicative.ofAdd (Neg.neg d))
      triv : Not (Eq g f)
      hn : LT.lt n d
      ⊢ Eq ((HSub.hSub g f).coeff n) 0
    -/
    apply coeff_zero_of_lt_valuation K H hn
    /-
      🎉 no goals
    -/

/- Every Laurent series of valuation less than `(1 : ℤₘ₀)` comes from a power series. -/

theorem val_le_one_iff_eq_coe (f : K⸨X⸩) : Valued.v f ≤ (1 : ℤₘ₀) ↔
    ∃ F : K⟦X⟧, F = f := by
  /-
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    ⊢ Iff (LE.le (Valued.v f) 1) (Exists fun F => Eq ((HahnSeries.ofPowerSeries In …
  -/
  rw [← WithZero.coe_one, ← ofAdd_zero, ← neg_zero, valuation_le_iff_coeff_lt_eq_zero]
  /-
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    ⊢ Iff (∀ (n : Int), LT.lt n 0 → Eq (f.coeff n) 0) (Exists fun F => Eq ((HahnSe …
  -/
  refine ⟨fun h => ⟨PowerSeries.mk fun n => f.coeff n, ?_⟩, ?_⟩
  /-
    case refine_1
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    h : ∀ (n : Int), LT.lt n 0 → Eq (f.coeff n) 0
    ⊢ Eq ((HahnSeries.ofPowerSeries Int K) (PowerSeries.mk fun n => f.coeff ↑n)) f
  -/
  on_goal 1 => ext (_ | n)
    /-
      case refine_1.coeff.h.ofNat
      K : Type u_2
      inst✝ : Field K
      f : LaurentSeries K
      h : ∀ (n : Int), LT.lt n 0 → Eq (f.coeff n) 0
      a✝ : Nat
      ⊢ Eq (((HahnSeries.ofPowerSeries Int K) (PowerSeries.mk fun n => f.coeff ↑n)). …
    -/
  · simp only [Int.ofNat_eq_coe, coeff_coe_powerSeries, coeff_mk]
    /-
      🎉 no goals
    -/
  /-
    case refine_1.coeff.h.negSucc
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    h : ∀ (n : Int), LT.lt n 0 → Eq (f.coeff n) 0
    n : Nat
    ⊢ Eq (((HahnSeries.ofPowerSeries Int K) (PowerSeries.mk fun n => f.coeff ↑n)). …
  -/
  on_goal 1 => simp only [h (Int.negSucc n) (Int.negSucc_lt_zero n)]
  /-
    case refine_1.coeff.h.negSucc
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    h : ∀ (n : Int), LT.lt n 0 → Eq (f.coeff n) 0
    n : Nat
    ⊢ Eq (((HahnSeries.ofPowerSeries Int K) (PowerSeries.mk fun n => f.coeff ↑n)). …
  -/
  on_goal 2 => rintro ⟨F, rfl⟩ _ _
  all_goals
    apply HahnSeries.embDomain_notin_range
    simp only [Nat.coe_castAddMonoidHom, RelEmbedding.coe_mk, Function.Embedding.coeFn_mk,
      Set.mem_range, not_exists, Int.negSucc_lt_zero, reduceCtorEq]
    intro
    /-
      case refine_1.coeff.h.negSucc.hb
      K : Type u_2
      inst✝ : Field K
      f : LaurentSeries K
      h : ∀ (n : Int), LT.lt n 0 → Eq (f.coeff n) 0
      n x✝ : Nat
      ⊢ Not False
    -/
  · simp only [not_false_eq_true]
    /-
      🎉 no goals
    -/
    /-
      case refine_2.intro.hb
      K : Type u_2
      inst✝ : Field K
      F : PowerSeries K
      n✝ : Int
      a✝ : LT.lt n✝ 0
      x✝ : Nat
      ⊢ Not (Eq (↑x✝) n✝)
    -/
  · omega
    /-
      🎉 no goals
    -/


theorem uniformContinuous_coeff {uK : UniformSpace K} (d : ℤ) :
    UniformContinuous fun f : K⸨X⸩ ↦ f.coeff d := by
  /-
    K : Type u_2
    inst✝ : Field K
    uK : UniformSpace K
    d : Int
    ⊢ UniformContinuous fun f => f.coeff d
  -/
  refine uniformContinuous_iff_eventually.mpr fun S hS ↦ eventually_iff_exists_mem.mpr ?_
  /-
    K : Type u_2
    inst✝ : Field K
    uK : UniformSpace K
    d : Int
    S : Set (Prod K K)
    hS : Membership.mem (uniformity K) S
    ⊢ Exists fun v => And (Membership.mem (uniformity (LaurentSeries K)) v) (∀ (y  …
  -/
  let γ : ℤₘ₀ˣ := Units.mk0 (↑(Multiplicative.ofAdd (-(d + 1)))) WithZero.coe_ne_zero
  /-
    K : Type u_2
    inst✝ : Field K
    uK : UniformSpace K
    d : Int
    S : Set (Prod K K)
    hS : Membership.mem (uniformity K) S
    γ : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
    ⊢ Exists fun v => And (Membership.mem (uniformity (LaurentSeries K)) v) (∀ (y  …
  -/
  use {P | Valued.v (P.snd - P.fst) < ↑γ}
  /-
    case h
    K : Type u_2
    inst✝ : Field K
    uK : UniformSpace K
    d : Int
    S : Set (Prod K K)
    hS : Membership.mem (uniformity K) S
    γ : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
    ⊢ And (Membership.mem (uniformity (LaurentSeries K)) (setOf fun P => LT.lt (Va …
  -/
  refine ⟨(Valued.hasBasis_uniformity K⸨X⸩ ℤₘ₀).mem_of_mem (by tauto), fun P hP ↦ ?_⟩
  /-
    case h
    K : Type u_2
    inst✝ : Field K
    uK : UniformSpace K
    d : Int
    S : Set (Prod K K)
    hS : Membership.mem (uniformity K) S
    γ : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
    P : Prod (LaurentSeries K) (LaurentSeries K)
    hP : Membership.mem (setOf fun P => LT.lt (Valued.v (HSub.hSub P.2 P.1)) ↑γ) P
    ⊢ Membership.mem S { fst := P.1.coeff d, snd := P.2.coeff d }
  -/
  rw [eq_coeff_of_valuation_sub_lt K (le_of_lt hP) (lt_add_one _)]
  /-
    case h
    K : Type u_2
    inst✝ : Field K
    uK : UniformSpace K
    d : Int
    S : Set (Prod K K)
    hS : Membership.mem (uniformity K) S
    γ : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
    P : Prod (LaurentSeries K) (LaurentSeries K)
    hP : Membership.mem (setOf fun P => LT.lt (Valued.v (HSub.hSub P.2 P.1)) ↑γ) P
    ⊢ Membership.mem S { fst := P.1.coeff d, snd := P.1.coeff d }
  -/
  exact mem_uniformity_of_eq hS rfl
  /-
    🎉 no goals
  -/


/-- Since extracting coefficients is uniformly continuous, every Cauchy filter in
`K⸨X⸩` gives rise to a Cauchy filter in `K` for every `d : ℤ`, and such Cauchy filter
in `K` converges to a principal filter -/
def Cauchy.coeff {ℱ : Filter K⸨X⸩} (hℱ : Cauchy ℱ) : ℤ → K :=
  let _ : UniformSpace K := ⊥
  fun d ↦ UniformSpace.DiscreteUnif.cauchyConst rfl <| hℱ.map (uniformContinuous_coeff d)


theorem Cauchy.coeff_tendsto {ℱ : Filter K⸨X⸩} (hℱ : Cauchy ℱ) (D : ℤ) :
    Tendsto (fun f : K⸨X⸩ ↦ f.coeff D) ℱ (𝓟 {coeff hℱ D}) :=
  let _ : UniformSpace K := ⊥
                                                               /-
                                                                 K : Type u_2
                                                                 inst✝ : Field K
                                                                 ℱ : Filter (LaurentSeries K)
                                                                 hℱ : Cauchy ℱ
                                                                 D : Int
                                                                 x✝ : UniformSpace K := Bot.bot
                                                                 ⊢ Eq x✝ Bot.bot
                                                               -/
  le_of_eq <| UniformSpace.DiscreteUnif.eq_const_of_cauchy (by rfl)
                                                               /-
                                                                 🎉 no goals
                                                               -/
    (hℱ.map (uniformContinuous_coeff D)) ▸ (principal_singleton _).symm

/- For every Cauchy filter of Laurent series, there is a `N` such that the `n`-th coefficient
vanishes for all `n ≤ N` and almost all series in the filter. This is an auxiliary lemma used
to construct the limit of the Cauchy filter as a Laurent series, ensuring that the support of the
limit is `PWO`.
The result is true also for more general Hahn Series indexed over a partially ordered group `Γ`
beyond the special case `Γ = ℤ`, that corresponds to Laurent Series: nevertheless the proof below
does not generalise, as it relies on the study of the `X`-adic valuation attached to the height-one
prime `X`, and this is peculiar to the one-variable setting. In the future we should prove this
result in full generality and deduce the case `Γ = ℤ` from that one.-/

lemma Cauchy.exists_lb_eventual_support {ℱ : Filter K⸨X⸩} (hℱ : Cauchy ℱ) :
    ∃ N, ∀ᶠ f : K⸨X⸩ in ℱ, ∀ n < N, f.coeff n = (0 : K) := by
  let entourage : Set (K⸨X⸩ × K⸨X⸩) :=
    {P : K⸨X⸩ × K⸨X⸩ |
      Valued.v (P.snd - P.fst) < ((Multiplicative.ofAdd 0 : Multiplicative ℤ) : ℤₘ₀)}
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    entourage : Set (Prod (LaurentSeries K) (LaurentSeries K)) := setOf fun P => L …
    ⊢ Exists fun N => Filter.Eventually (fun f => ∀ (n : Int), LT.lt n N → Eq (f.c …
  -/
  let ζ := Units.mk0 (G₀ := ℤₘ₀) _ (WithZero.coe_ne_zero (a := (Multiplicative.ofAdd 0)))
  obtain ⟨S, ⟨hS, ⟨T, ⟨hT, H⟩⟩⟩⟩ := mem_prod_iff.mp <| Filter.le_def.mp hℱ.2 entourage
    <| (Valued.hasBasis_uniformity K⸨X⸩ ℤₘ₀).mem_of_mem (i := ζ) (by tauto)
  /-
    case intro.intro.intro.intro
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    entourage : Set (Prod (LaurentSeries K) (LaurentSeries K)) := setOf fun P => L …
    ζ : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
    S : Set (LaurentSeries K)
    hS : Membership.mem ℱ S
    T : Set (LaurentSeries K)
    hT : Membership.mem ℱ T
    H : HasSubset.Subset (SProd.sprod S T) entourage
    ⊢ Exists fun N => Filter.Eventually (fun f => ∀ (n : Int), LT.lt n N → Eq (f.c …
  -/
  obtain ⟨f, hf⟩ := forall_mem_nonempty_iff_neBot.mpr hℱ.1 (S ∩ T) (inter_mem_iff.mpr ⟨hS, hT⟩)
  obtain ⟨N, hN⟩ :  ∃ N : ℤ, ∀ g : K⸨X⸩,
    Valued.v (g - f) ≤ ↑(Multiplicative.ofAdd (0 : ℤ)) → ∀ n < N, g.coeff n = 0 := by
    by_cases hf : f = 0
    · refine ⟨0, fun x hg ↦ ?_⟩
      rw [hf, sub_zero] at hg
      exact (valuation_le_iff_coeff_lt_eq_zero K).mp hg
    · refine ⟨min (f.2.isWF.min (HahnSeries.support_nonempty_iff.mpr hf)) 0 - 1, fun _ hg n hn ↦ ?_⟩
      rw [eq_coeff_of_valuation_sub_lt K hg (d := 0)]
      · exact Function.nmem_support.mp fun h ↦
        f.2.isWF.not_lt_min (HahnSeries.support_nonempty_iff.mpr hf) h
        <| lt_trans hn <| Int.sub_one_lt_iff.mpr <| min_le_left _ _
      exact lt_of_lt_of_le hn <| le_of_lt (Int.sub_one_lt_of_le <| min_le_right _ _)
  /-
    case intro.intro.intro.intro.intro.intro
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    entourage : Set (Prod (LaurentSeries K) (LaurentSeries K)) := setOf fun P => L …
    ζ : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
    S : Set (LaurentSeries K)
    hS : Membership.mem ℱ S
    T : Set (LaurentSeries K)
    hT : Membership.mem ℱ T
    H : HasSubset.Subset (SProd.sprod S T) entourage
    f : LaurentSeries K
    hf : Membership.mem (Inter.inter S T) f
    N : Int
    hN : ∀ (g : LaurentSeries K), LE.le (Valued.v (HSub.hSub g f)) ↑(Multiplicativ …
    ⊢ Exists fun N => Filter.Eventually (fun f => ∀ (n : Int), LT.lt n N → Eq (f.c …
  -/
  use N
  /-
    case h
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    entourage : Set (Prod (LaurentSeries K) (LaurentSeries K)) := setOf fun P => L …
    ζ : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
    S : Set (LaurentSeries K)
    hS : Membership.mem ℱ S
    T : Set (LaurentSeries K)
    hT : Membership.mem ℱ T
    H : HasSubset.Subset (SProd.sprod S T) entourage
    f : LaurentSeries K
    hf : Membership.mem (Inter.inter S T) f
    N : Int
    hN : ∀ (g : LaurentSeries K), LE.le (Valued.v (HSub.hSub g f)) ↑(Multiplicativ …
    ⊢ Filter.Eventually (fun f => ∀ (n : Int), LT.lt n N → Eq (f.coeff n) 0) ℱ
  -/
  apply mem_of_superset (inter_mem hS hT)
  /-
    case h
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    entourage : Set (Prod (LaurentSeries K) (LaurentSeries K)) := setOf fun P => L …
    ζ : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
    S : Set (LaurentSeries K)
    hS : Membership.mem ℱ S
    T : Set (LaurentSeries K)
    hT : Membership.mem ℱ T
    H : HasSubset.Subset (SProd.sprod S T) entourage
    f : LaurentSeries K
    hf : Membership.mem (Inter.inter S T) f
    N : Int
    hN : ∀ (g : LaurentSeries K), LE.le (Valued.v (HSub.hSub g f)) ↑(Multiplicativ …
    ⊢ HasSubset.Subset (Inter.inter S T) (setOf fun x => (fun f => ∀ (n : Int), LT …
  -/
  intro g hg
  have h_prod : (f, g) ∈ entourage := Set.prod_mono (Set.inter_subset_left (t := T))
    (Set.inter_subset_right (s := S)) |>.trans H <| Set.mem_prod.mpr ⟨hf, hg⟩
  /-
    case h
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    entourage : Set (Prod (LaurentSeries K) (LaurentSeries K)) := setOf fun P => L …
    ζ : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
    S : Set (LaurentSeries K)
    hS : Membership.mem ℱ S
    T : Set (LaurentSeries K)
    hT : Membership.mem ℱ T
    H : HasSubset.Subset (SProd.sprod S T) entourage
    f : LaurentSeries K
    hf : Membership.mem (Inter.inter S T) f
    N : Int
    hN : ∀ (g : LaurentSeries K), LE.le (Valued.v (HSub.hSub g f)) ↑(Multiplicativ …
    g : LaurentSeries K
    hg : Membership.mem (Inter.inter S T) g
    h_prod : Membership.mem entourage { fst := f, snd := g }
    ⊢ Membership.mem (setOf fun x => (fun f => ∀ (n : Int), LT.lt n N → Eq (f.coef …
  -/
  exact hN g (le_of_lt h_prod)
  /-
    🎉 no goals
  -/

/- The support of `Cauchy.coeff` has a lower bound. -/

theorem Cauchy.exists_lb_support {ℱ : Filter K⸨X⸩} (hℱ : Cauchy ℱ) :
    ∃ N, ∀ n, n < N → coeff hℱ n = 0 := by
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    ⊢ Exists fun N => ∀ (n : Int), LT.lt n N → Eq (LaurentSeries.Cauchy.coeff hℱ n …
  -/
  let _ : UniformSpace K := ⊥
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    x✝ : UniformSpace K := Bot.bot
    ⊢ Exists fun N => ∀ (n : Int), LT.lt n N → Eq (LaurentSeries.Cauchy.coeff hℱ n …
  -/
  obtain ⟨N, hN⟩ := exists_lb_eventual_support hℱ
  refine ⟨N, fun n hn ↦ Ultrafilter.eq_of_le_pure (hℱ.map (uniformContinuous_coeff n)).1
      ((principal_singleton _).symm ▸ coeff_tendsto _ _) ?_⟩
  /-
    case intro
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    x✝ : UniformSpace K := Bot.bot
    N : Int
    hN : Filter.Eventually (fun f => ∀ (n : Int), LT.lt n N → Eq (f.coeff n) 0) ℱ
    n : Int
    hn : LT.lt n N
    ⊢ LE.le (Filter.map (fun f => f.coeff n) ℱ) (Pure.pure 0)
  -/
  simp only [pure_zero, nonpos_iff]
  /-
    case intro
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    x✝ : UniformSpace K := Bot.bot
    N : Int
    hN : Filter.Eventually (fun f => ∀ (n : Int), LT.lt n N → Eq (f.coeff n) 0) ℱ
    n : Int
    hn : LT.lt n N
    ⊢ Membership.mem (Filter.map (fun f => f.coeff n) ℱ) 0
  -/
  apply Filter.mem_of_superset hN (fun _ ha ↦ ha _ hn)
  /-
    🎉 no goals
  -/

/- The support of `Cauchy.coeff` is bounded below -/

theorem Cauchy.coeff_support_bddBelow {ℱ : Filter K⸨X⸩} (hℱ : Cauchy ℱ) :
    BddBelow (coeff hℱ).support := by
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    ⊢ BddBelow (Function.support (LaurentSeries.Cauchy.coeff hℱ))
  -/
  refine ⟨(exists_lb_support hℱ).choose, fun d hd ↦ ?_⟩
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    d : Int
    hd : Membership.mem (Function.support (LaurentSeries.Cauchy.coeff hℱ)) d
    ⊢ LE.le ⋯.choose d
  -/
  by_contra hNd
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    d : Int
    hd : Membership.mem (Function.support (LaurentSeries.Cauchy.coeff hℱ)) d
    hNd : Not (LE.le ⋯.choose d)
    ⊢ False
  -/
  exact hd ((exists_lb_support hℱ).choose_spec d (not_le.mp hNd))
  /-
    🎉 no goals
  -/


/-- To any Cauchy filter ℱ of `K⸨X⸩`, we can attach a laurent series that is the limit
of the filter. Its `d`-th coefficient is defined as the limit of `Cauchy.coeff hℱ d`, which is
again Cauchy but valued in the discrete space `K`. That sufficiently negative coefficients vanish
follows from `Cauchy.coeff_support_bddBelow` -/
def Cauchy.limit {ℱ : Filter K⸨X⸩} (hℱ : Cauchy ℱ) : K⸨X⸩ :=
  HahnSeries.mk (coeff hℱ) <| Set.IsWF.isPWO (coeff_support_bddBelow _).wellFoundedOn_lt

/- The following lemma shows that for every `d` smaller than the minimum between the integers
produced in `Cauchy.exists_lb_eventual_support` and `Cauchy.exists_lb_support`, for almost all
series in `ℱ` the `d`th coefficient coincides with the `d`th coefficient of `Cauchy.coeff hℱ`. -/

theorem Cauchy.exists_lb_coeff_ne {ℱ : Filter K⸨X⸩} (hℱ : Cauchy ℱ) :
    ∃ N, ∀ᶠ f : K⸨X⸩ in ℱ, ∀ d < N, coeff hℱ d = f.coeff d := by
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    ⊢ Exists fun N => Filter.Eventually (fun f => ∀ (d : Int), LT.lt d N → Eq (Lau …
  -/
  obtain ⟨⟨N₁, hN₁⟩, ⟨N₂, hN₂⟩⟩ := exists_lb_eventual_support hℱ, exists_lb_support hℱ
  /-
    case intro.intro
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    N₁ : Int
    hN₁ : Filter.Eventually (fun f => ∀ (n : Int), LT.lt n N₁ → Eq (f.coeff n) 0) ℱ
    N₂ : Int
    hN₂ : ∀ (n : Int), LT.lt n N₂ → Eq (LaurentSeries.Cauchy.coeff hℱ n) 0
    ⊢ Exists fun N => Filter.Eventually (fun f => ∀ (d : Int), LT.lt d N → Eq (Lau …
  -/
  refine ⟨min N₁ N₂, ℱ.3 hN₁ fun _ hf d hd ↦ ?_⟩
  /-
    case intro.intro
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    N₁ : Int
    hN₁ : Filter.Eventually (fun f => ∀ (n : Int), LT.lt n N₁ → Eq (f.coeff n) 0) ℱ
    N₂ : Int
    hN₂ : ∀ (n : Int), LT.lt n N₂ → Eq (LaurentSeries.Cauchy.coeff hℱ n) 0
    x✝ : LaurentSeries K
    hf : Membership.mem (setOf fun x => (fun f => ∀ (n : Int), LT.lt n N₁ → Eq (f. …
    d : Int
    hd : LT.lt d (Min.min N₁ N₂)
    ⊢ Eq (LaurentSeries.Cauchy.coeff hℱ d) (x✝.coeff d)
  -/
  rw [hf d (lt_of_lt_of_le hd (min_le_left _ _)), hN₂ d (lt_of_lt_of_le hd (min_le_right _ _))]
  /-
    🎉 no goals
  -/

/- Given a Cauchy filter `ℱ` in the Laurent Series and a bound `D`, for almost all series in the
filter the coefficients below `D` coincide with `Caucy.coeff hℱ`-/

theorem Cauchy.coeff_eventually_equal {ℱ : Filter K⸨X⸩} (hℱ : Cauchy ℱ) {D : ℤ} :
    ∀ᶠ f : K⸨X⸩ in ℱ, ∀ d, d < D → coeff hℱ d = f.coeff d := by
  -- `φ` sends `d` to the set of Laurent Series having `d`th coefficient equal to `ℱ.coeff`.
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    D : Int
    ⊢ Filter.Eventually (fun f => ∀ (d : Int), LT.lt d D → Eq (LaurentSeries.Cauch …
  -/
  let φ : ℤ → Set K⸨X⸩ := fun d ↦ {f | coeff hℱ d = f.coeff d}
  have intersec₁ :
    (⋂ n ∈ Set.Iio D, φ n) ⊆ {x : K⸨X⸩ | ∀ d : ℤ, d < D → coeff hℱ d = x.coeff d} := by
    intro _ hf
    simpa only [Set.mem_iInter] using hf
  -- The goal is now to show that the intersection of all `φ d` (for `d < D`) is in `ℱ`.
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    D : Int
    φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
    intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
    ⊢ Filter.Eventually (fun f => ∀ (d : Int), LT.lt d D → Eq (LaurentSeries.Cauch …
  -/
  let ℓ := (exists_lb_coeff_ne hℱ).choose
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    D : Int
    φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
    intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
    ℓ : Int := ⋯.choose
    ⊢ Filter.Eventually (fun f => ∀ (d : Int), LT.lt d D → Eq (LaurentSeries.Cauch …
  -/
  let N := max ℓ D
  have intersec₂ : ⋂ n ∈ Set.Iio D, φ n ⊇ (⋂ n ∈ Set.Iio ℓ, φ n) ∩ (⋂ n ∈ Set.Icc ℓ N, φ n) := by
    simp only [Set.mem_Iio, Set.mem_Icc, Set.subset_iInter_iff]
    intro i hi x hx
    simp only [Set.mem_inter_iff, Set.mem_iInter, and_imp] at hx
    by_cases H : i < ℓ
    exacts [hx.1 _ H, hx.2 _ (le_of_not_lt H) <| le_of_lt <| lt_max_of_lt_right hi]
  suffices (⋂ n ∈ Set.Iio ℓ, φ n) ∩ (⋂ n ∈ Set.Icc ℓ N, φ n) ∈ ℱ by
    exact ℱ.sets_of_superset this <| intersec₂.trans intersec₁
  /- To show that the intersection we have in sight is in `ℱ`, we use that it contains a double
  intersection (an infinite and a finite one): by general properties of filters, we are reduced
  to show that both terms are in `ℱ`, which is easy in light of their definition. -/
    /-
      K : Type u_2
      inst✝ : Field K
      ℱ : Filter (LaurentSeries K)
      hℱ : Cauchy ℱ
      D : Int
      φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
      intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
      ℓ : Int := ⋯.choose
      N : Int := Max.max ℓ D
      intersec₂ : Superset (Set.iInter fun n => Set.iInter fun h => φ n) (Inter.inte …
      ⊢ Membership.mem ℱ (Inter.inter (Set.iInter fun n => Set.iInter fun h => φ n)  …
    -/
  · simp only [Set.mem_Iio, Set.mem_Ico, inter_mem_iff]
    /-
      K : Type u_2
      inst✝ : Field K
      ℱ : Filter (LaurentSeries K)
      hℱ : Cauchy ℱ
      D : Int
      φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
      intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
      ℓ : Int := ⋯.choose
      N : Int := Max.max ℓ D
      intersec₂ : Superset (Set.iInter fun n => Set.iInter fun h => φ n) (Inter.inte …
      ⊢ And (Membership.mem ℱ (Set.iInter fun n => Set.iInter fun x => φ n)) (Member …
    -/
    constructor
      /-
        case left
        K : Type u_2
        inst✝ : Field K
        ℱ : Filter (LaurentSeries K)
        hℱ : Cauchy ℱ
        D : Int
        φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
        intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
        ℓ : Int := ⋯.choose
        N : Int := Max.max ℓ D
        intersec₂ : Superset (Set.iInter fun n => Set.iInter fun h => φ n) (Inter.inte …
        ⊢ Membership.mem ℱ (Set.iInter fun n => Set.iInter fun x => φ n)
      -/
    · have := (exists_lb_coeff_ne hℱ).choose_spec
      /-
        case left
        K : Type u_2
        inst✝ : Field K
        ℱ : Filter (LaurentSeries K)
        hℱ : Cauchy ℱ
        D : Int
        φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
        intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
        ℓ : Int := ⋯.choose
        N : Int := Max.max ℓ D
        intersec₂ : Superset (Set.iInter fun n => Set.iInter fun h => φ n) (Inter.inte …
        this : Filter.Eventually (fun f => ∀ (d : Int), LT.lt d ⋯.choose → Eq (Laurent …
        ⊢ Membership.mem ℱ (Set.iInter fun n => Set.iInter fun x => φ n)
      -/
      rw [Filter.eventually_iff] at this
      /-
        case left
        K : Type u_2
        inst✝ : Field K
        ℱ : Filter (LaurentSeries K)
        hℱ : Cauchy ℱ
        D : Int
        φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
        intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
        ℓ : Int := ⋯.choose
        N : Int := Max.max ℓ D
        intersec₂ : Superset (Set.iInter fun n => Set.iInter fun h => φ n) (Inter.inte …
        this : Membership.mem ℱ (setOf fun x => ∀ (d : Int), LT.lt d ⋯.choose → Eq (La …
        ⊢ Membership.mem ℱ (Set.iInter fun n => Set.iInter fun x => φ n)
      -/
      convert this
      /-
        case h.e'_5
        K : Type u_2
        inst✝ : Field K
        ℱ : Filter (LaurentSeries K)
        hℱ : Cauchy ℱ
        D : Int
        φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
        intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
        ℓ : Int := ⋯.choose
        N : Int := Max.max ℓ D
        intersec₂ : Superset (Set.iInter fun n => Set.iInter fun h => φ n) (Inter.inte …
        this : Membership.mem ℱ (setOf fun x => ∀ (d : Int), LT.lt d ⋯.choose → Eq (La …
        ⊢ Eq (Set.iInter fun n => Set.iInter fun x => φ n) (setOf fun x => ∀ (d : Int) …
      -/
      ext
      /-
        case h.e'_5.h
        K : Type u_2
        inst✝ : Field K
        ℱ : Filter (LaurentSeries K)
        hℱ : Cauchy ℱ
        D : Int
        φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
        intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
        ℓ : Int := ⋯.choose
        N : Int := Max.max ℓ D
        intersec₂ : Superset (Set.iInter fun n => Set.iInter fun h => φ n) (Inter.inte …
        this : Membership.mem ℱ (setOf fun x => ∀ (d : Int), LT.lt d ⋯.choose → Eq (La …
        x✝ : LaurentSeries K
        ⊢ Iff (Membership.mem (Set.iInter fun n => Set.iInter fun x => φ n) x✝) (Membe …
      -/
      simp only [Set.mem_iInter, Set.mem_setOf_eq]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/
      /-
        case right
        K : Type u_2
        inst✝ : Field K
        ℱ : Filter (LaurentSeries K)
        hℱ : Cauchy ℱ
        D : Int
        φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
        intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
        ℓ : Int := ⋯.choose
        N : Int := Max.max ℓ D
        intersec₂ : Superset (Set.iInter fun n => Set.iInter fun h => φ n) (Inter.inte …
        ⊢ Membership.mem ℱ (Set.iInter fun n => Set.iInter fun h => φ n)
      -/
    · rw [biInter_mem (Set.finite_Icc ℓ N)]
      /-
        case right
        K : Type u_2
        inst✝ : Field K
        ℱ : Filter (LaurentSeries K)
        hℱ : Cauchy ℱ
        D : Int
        φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
        intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
        ℓ : Int := ⋯.choose
        N : Int := Max.max ℓ D
        intersec₂ : Superset (Set.iInter fun n => Set.iInter fun h => φ n) (Inter.inte …
        ⊢ ∀ (i : Int), Membership.mem (Set.Icc ℓ N) i → Membership.mem ℱ (φ i)
      -/
      intro _ _
      /-
        case right
        K : Type u_2
        inst✝ : Field K
        ℱ : Filter (LaurentSeries K)
        hℱ : Cauchy ℱ
        D : Int
        φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
        intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
        ℓ : Int := ⋯.choose
        N : Int := Max.max ℓ D
        intersec₂ : Superset (Set.iInter fun n => Set.iInter fun h => φ n) (Inter.inte …
        i✝ : Int
        a✝ : Membership.mem (Set.Icc ℓ N) i✝
        ⊢ Membership.mem ℱ (φ i✝)
      -/
      apply coeff_tendsto hℱ
      /-
        case right.a
        K : Type u_2
        inst✝ : Field K
        ℱ : Filter (LaurentSeries K)
        hℱ : Cauchy ℱ
        D : Int
        φ : Int → Set (LaurentSeries K) := fun d => setOf fun f => Eq (LaurentSeries.C …
        intersec₁ : HasSubset.Subset (Set.iInter fun n => Set.iInter fun h => φ n) (se …
        ℓ : Int := ⋯.choose
        N : Int := Max.max ℓ D
        intersec₂ : Superset (Set.iInter fun n => Set.iInter fun h => φ n) (Inter.inte …
        i✝ : Int
        a✝ : Membership.mem (Set.Icc ℓ N) i✝
        ⊢ Membership.mem (Filter.principal (Singleton.singleton (LaurentSeries.Cauchy. …
      -/
      simp only [principal_singleton, mem_pure]; rfl
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem Cauchy.eventually_mem_nhds {ℱ : Filter K⸨X⸩} (hℱ : Cauchy ℱ)
    {U : Set K⸨X⸩} (hU : U ∈ 𝓝 (Cauchy.limit hℱ)) : ∀ᶠ f in ℱ, f ∈ U := by
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    U : Set (LaurentSeries K)
    hU : Membership.mem (nhds (LaurentSeries.Cauchy.limit hℱ)) U
    ⊢ Filter.Eventually (fun f => Membership.mem U f) ℱ
  -/
  obtain ⟨γ, hU₁⟩ := Valued.mem_nhds.mp hU
  suffices ∀ᶠ f in ℱ, f ∈ {y : K⸨X⸩ | Valued.v (y - limit hℱ) < ↑γ} by
    apply this.mono fun _ hf ↦ hU₁ hf
  /-
    case intro
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    U : Set (LaurentSeries K)
    hU : Membership.mem (nhds (LaurentSeries.Cauchy.limit hℱ)) U
    γ : Units (WithZero (Multiplicative Int))
    hU₁ : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (LaurentSe …
    ⊢ Filter.Eventually (fun f => Membership.mem (setOf fun y => LT.lt (Valued.v ( …
  -/
  set D := -((WithZero.unzero γ.ne_zero).toAdd - 1) with hD₀
  have hD : ((Multiplicative.ofAdd (-D) : Multiplicative ℤ) : ℤₘ₀) < γ := by
    rw [← WithZero.coe_unzero γ.ne_zero, WithZero.coe_lt_coe, hD₀, neg_neg, ofAdd_sub,
      ofAdd_toAdd, div_lt_comm, div_self', ← ofAdd_zero, Multiplicative.ofAdd_lt]
    exact zero_lt_one
  /-
    case intro
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    U : Set (LaurentSeries K)
    hU : Membership.mem (nhds (LaurentSeries.Cauchy.limit hℱ)) U
    γ : Units (WithZero (Multiplicative Int))
    hU₁ : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (LaurentSe …
    D : Int := Neg.neg (HSub.hSub (Multiplicative.toAdd (WithZero.unzero ⋯)) 1)
    hD₀ : Eq D (Neg.neg (HSub.hSub (Multiplicative.toAdd (WithZero.unzero ⋯)) 1))
    hD : LT.lt ↑(Multiplicative.ofAdd (Neg.neg D)) ↑γ
    ⊢ Filter.Eventually (fun f => Membership.mem (setOf fun y => LT.lt (Valued.v ( …
  -/
  apply coeff_eventually_equal (D := D) hℱ |>.mono
  /-
    case intro
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    U : Set (LaurentSeries K)
    hU : Membership.mem (nhds (LaurentSeries.Cauchy.limit hℱ)) U
    γ : Units (WithZero (Multiplicative Int))
    hU₁ : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (LaurentSe …
    D : Int := Neg.neg (HSub.hSub (Multiplicative.toAdd (WithZero.unzero ⋯)) 1)
    hD₀ : Eq D (Neg.neg (HSub.hSub (Multiplicative.toAdd (WithZero.unzero ⋯)) 1))
    hD : LT.lt ↑(Multiplicative.ofAdd (Neg.neg D)) ↑γ
    ⊢ ∀ (x : LaurentSeries K), (∀ (d : Int), LT.lt d D → Eq (LaurentSeries.Cauchy. …
  -/
  intro _ hf
  /-
    case intro
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    U : Set (LaurentSeries K)
    hU : Membership.mem (nhds (LaurentSeries.Cauchy.limit hℱ)) U
    γ : Units (WithZero (Multiplicative Int))
    hU₁ : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (LaurentSe …
    D : Int := Neg.neg (HSub.hSub (Multiplicative.toAdd (WithZero.unzero ⋯)) 1)
    hD₀ : Eq D (Neg.neg (HSub.hSub (Multiplicative.toAdd (WithZero.unzero ⋯)) 1))
    hD : LT.lt ↑(Multiplicative.ofAdd (Neg.neg D)) ↑γ
    x✝ : LaurentSeries K
    hf : ∀ (d : Int), LT.lt d D → Eq (LaurentSeries.Cauchy.coeff hℱ d) (x✝.coeff d)
    ⊢ Membership.mem (setOf fun y => LT.lt (Valued.v (HSub.hSub y (LaurentSeries.C …
  -/
  apply lt_of_le_of_lt (valuation_le_iff_coeff_lt_eq_zero K |>.mpr _) hD
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    U : Set (LaurentSeries K)
    hU : Membership.mem (nhds (LaurentSeries.Cauchy.limit hℱ)) U
    γ : Units (WithZero (Multiplicative Int))
    hU₁ : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (LaurentSe …
    D : Int := Neg.neg (HSub.hSub (Multiplicative.toAdd (WithZero.unzero ⋯)) 1)
    hD₀ : Eq D (Neg.neg (HSub.hSub (Multiplicative.toAdd (WithZero.unzero ⋯)) 1))
    hD : LT.lt ↑(Multiplicative.ofAdd (Neg.neg D)) ↑γ
    x✝ : LaurentSeries K
    hf : ∀ (d : Int), LT.lt d D → Eq (LaurentSeries.Cauchy.coeff hℱ d) (x✝.coeff d)
    ⊢ ∀ (n : Int), LT.lt n D → Eq ((HSub.hSub x✝ (LaurentSeries.Cauchy.limit hℱ)). …
  -/
  intro n hn
  /-
    K : Type u_2
    inst✝ : Field K
    ℱ : Filter (LaurentSeries K)
    hℱ : Cauchy ℱ
    U : Set (LaurentSeries K)
    hU : Membership.mem (nhds (LaurentSeries.Cauchy.limit hℱ)) U
    γ : Units (WithZero (Multiplicative Int))
    hU₁ : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y (LaurentSe …
    D : Int := Neg.neg (HSub.hSub (Multiplicative.toAdd (WithZero.unzero ⋯)) 1)
    hD₀ : Eq D (Neg.neg (HSub.hSub (Multiplicative.toAdd (WithZero.unzero ⋯)) 1))
    hD : LT.lt ↑(Multiplicative.ofAdd (Neg.neg D)) ↑γ
    x✝ : LaurentSeries K
    hf : ∀ (d : Int), LT.lt d D → Eq (LaurentSeries.Cauchy.coeff hℱ d) (x✝.coeff d)
    n : Int
    hn : LT.lt n D
    ⊢ Eq ((HSub.hSub x✝ (LaurentSeries.Cauchy.limit hℱ)).coeff n) 0
  -/
  rw [HahnSeries.sub_coeff, sub_eq_zero, hf n hn |>.symm]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/

/- Laurent Series with coefficients in a field are complete w.r.t. the `X`-adic valuation -/

instance instLaurentSeriesComplete : CompleteSpace K⸨X⸩ :=
  ⟨fun hℱ ↦ ⟨Cauchy.limit hℱ, fun _ hS ↦ Cauchy.eventually_mem_nhds hℱ hS⟩⟩


theorem exists_Polynomial_intValuation_lt (F : K⟦X⟧) (η : ℤₘ₀ˣ) :
    ∃ P : K[X], (PowerSeries.idealX K).intValuation (F - P) < η := by
  /-
    K : Type u_2
    inst✝ : Field K
    F : PowerSeries K
    η : Units (WithZero (Multiplicative Int))
    ⊢ Exists fun P => LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑P)) …
  -/
  by_cases h_neg : 1 < η
    /-
      case pos
      K : Type u_2
      inst✝ : Field K
      F : PowerSeries K
      η : Units (WithZero (Multiplicative Int))
      h_neg : LT.lt 1 η
      ⊢ Exists fun P => LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑P)) …
    -/
  · use 0
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      F : PowerSeries K
      η : Units (WithZero (Multiplicative Int))
      h_neg : LT.lt 1 η
      ⊢ LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑0)) ↑η
    -/
    simpa using (intValuation_le_one (PowerSeries.idealX K) F).trans_lt h_neg
    /-
      🎉 no goals
    -/
  · rw [not_lt, ← Units.val_le_val, Units.val_one, ← WithZero.coe_one, ← coe_unzero η.ne_zero,
      coe_le_coe, ← Multiplicative.toAdd_le, toAdd_one] at h_neg
    /-
      case neg
      K : Type u_2
      inst✝ : Field K
      F : PowerSeries K
      η : Units (WithZero (Multiplicative Int))
      h_neg : LE.le (Multiplicative.toAdd (WithZero.unzero ⋯)) 0
      ⊢ Exists fun P => LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑P)) …
    -/
    obtain ⟨d, hd⟩ := Int.exists_eq_neg_ofNat h_neg
    /-
      case neg.intro
      K : Type u_2
      inst✝ : Field K
      F : PowerSeries K
      η : Units (WithZero (Multiplicative Int))
      h_neg : LE.le (Multiplicative.toAdd (WithZero.unzero ⋯)) 0
      d : Nat
      hd : Eq (Multiplicative.toAdd (WithZero.unzero ⋯)) (Neg.neg ↑d)
      ⊢ Exists fun P => LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑P)) …
    -/
    use F.trunc (d + 1)
    have : Valued.v ((ofPowerSeries ℤ K) (F - (trunc (d + 1) F))) ≤
      (Multiplicative.ofAdd (-(d + 1 : ℤ))) := by
      apply (intValuation_le_iff_coeff_lt_eq_zero K _).mpr
      simpa only [map_sub, sub_eq_zero, Polynomial.coeff_coe, coeff_trunc] using
        fun _ h ↦ (if_pos h).symm
    rw [ neg_add, ofAdd_add, ← hd, ofAdd_toAdd, WithZero.coe_mul, coe_unzero,
      ← coe_algebraMap] at this
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      F : PowerSeries K
      η : Units (WithZero (Multiplicative Int))
      h_neg : LE.le (Multiplicative.toAdd (WithZero.unzero ⋯)) 0
      d : Nat
      hd : Eq (Multiplicative.toAdd (WithZero.unzero ⋯)) (Neg.neg ↑d)
      this : LE.le (Valued.v ((algebraMap (PowerSeries K) (LaurentSeries K)) (HSub.h …
      ⊢ LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑(PowerSeries.trunc  …
    -/
    rw [← valuation_of_algebraMap (K := K⸨X⸩) (PowerSeries.idealX K) (F - F.trunc (d + 1))]
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      F : PowerSeries K
      η : Units (WithZero (Multiplicative Int))
      h_neg : LE.le (Multiplicative.toAdd (WithZero.unzero ⋯)) 0
      d : Nat
      hd : Eq (Multiplicative.toAdd (WithZero.unzero ⋯)) (Neg.neg ↑d)
      this : LE.le (Valued.v ((algebraMap (PowerSeries K) (LaurentSeries K)) (HSub.h …
      ⊢ LT.lt ((PowerSeries.idealX K).valuation ((algebraMap (PowerSeries K) (Lauren …
    -/
    apply lt_of_le_of_lt this
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      F : PowerSeries K
      η : Units (WithZero (Multiplicative Int))
      h_neg : LE.le (Multiplicative.toAdd (WithZero.unzero ⋯)) 0
      d : Nat
      hd : Eq (Multiplicative.toAdd (WithZero.unzero ⋯)) (Neg.neg ↑d)
      this : LE.le (Valued.v ((algebraMap (PowerSeries K) (LaurentSeries K)) (HSub.h …
      ⊢ LT.lt (HMul.hMul ↑η ↑(Multiplicative.ofAdd (-1))) ↑η
    -/
    rw [← mul_one (η : ℤₘ₀), mul_assoc, one_mul]
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      F : PowerSeries K
      η : Units (WithZero (Multiplicative Int))
      h_neg : LE.le (Multiplicative.toAdd (WithZero.unzero ⋯)) 0
      d : Nat
      hd : Eq (Multiplicative.toAdd (WithZero.unzero ⋯)) (Neg.neg ↑d)
      this : LE.le (Valued.v ((algebraMap (PowerSeries K) (LaurentSeries K)) (HSub.h …
      ⊢ LT.lt (HMul.hMul ↑η ↑(Multiplicative.ofAdd (-1))) (HMul.hMul (↑η) 1)
    -/
    gcongr
      /-
        case h.a0
        K : Type u_2
        inst✝ : Field K
        F : PowerSeries K
        η : Units (WithZero (Multiplicative Int))
        h_neg : LE.le (Multiplicative.toAdd (WithZero.unzero ⋯)) 0
        d : Nat
        hd : Eq (Multiplicative.toAdd (WithZero.unzero ⋯)) (Neg.neg ↑d)
        this : LE.le (Valued.v ((algebraMap (PowerSeries K) (LaurentSeries K)) (HSub.h …
        ⊢ LT.lt 0 ↑η
      -/
    · exact zero_lt_iff.2 η.ne_zero
      /-
        🎉 no goals
      -/
    rw [← WithZero.coe_one, coe_lt_coe, ofAdd_neg, Right.inv_lt_one_iff, ← ofAdd_zero,
      Multiplicative.ofAdd_lt]
    /-
      case h.bc
      K : Type u_2
      inst✝ : Field K
      F : PowerSeries K
      η : Units (WithZero (Multiplicative Int))
      h_neg : LE.le (Multiplicative.toAdd (WithZero.unzero ⋯)) 0
      d : Nat
      hd : Eq (Multiplicative.toAdd (WithZero.unzero ⋯)) (Neg.neg ↑d)
      this : LE.le (Valued.v ((algebraMap (PowerSeries K) (LaurentSeries K)) (HSub.h …
      ⊢ LT.lt 0 1
    -/
    exact Int.zero_lt_one
    /-
      🎉 no goals
    -/


/-- For every Laurent series `f` and every `γ : ℤₘ₀` one can find a rational function `Q` such
that the `X`-adic valuation `v` satisfies `v (f - Q) < γ`. -/
theorem exists_ratFunc_val_lt (f : K⸨X⸩) (γ : ℤₘ₀ˣ) :
    ∃ Q : RatFunc K, Valued.v (f - Q) < γ := by
  /-
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    γ : Units (WithZero (Multiplicative Int))
    ⊢ Exists fun Q => LT.lt (Valued.v (HSub.hSub f ↑Q)) ↑γ
  -/
  set F := f.powerSeriesPart with hF
  /-
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    γ : Units (WithZero (Multiplicative Int))
    F : PowerSeries K := f.powerSeriesPart
    hF : Eq F f.powerSeriesPart
    ⊢ Exists fun Q => LT.lt (Valued.v (HSub.hSub f ↑Q)) ↑γ
  -/
  by_cases ord_nonpos : f.order < 0
  · set η : ℤₘ₀ˣ := Units.mk0 (Multiplicative.ofAdd f.order : Multiplicative ℤ) coe_ne_zero
      with hη
    /-
      case pos
      K : Type u_2
      inst✝ : Field K
      f : LaurentSeries K
      γ : Units (WithZero (Multiplicative Int))
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LT.lt (HahnSeries.order f) 0
      η : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
      hη : Eq η (Units.mk0 ↑(Multiplicative.ofAdd (HahnSeries.order f)) ⋯)
      ⊢ Exists fun Q => LT.lt (Valued.v (HSub.hSub f ↑Q)) ↑γ
    -/
    obtain ⟨P, hP⟩ := exists_Polynomial_intValuation_lt F (η * γ)
    /-
      case pos.intro
      K : Type u_2
      inst✝ : Field K
      f : LaurentSeries K
      γ : Units (WithZero (Multiplicative Int))
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LT.lt (HahnSeries.order f) 0
      η : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
      hη : Eq η (Units.mk0 ↑(Multiplicative.ofAdd (HahnSeries.order f)) ⋯)
      P : Polynomial K
      hP : LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑P)) ↑(HMul.hMul  …
      ⊢ Exists fun Q => LT.lt (Valued.v (HSub.hSub f ↑Q)) ↑γ
    -/
    use RatFunc.X ^ f.order * (P : RatFunc K)
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      f : LaurentSeries K
      γ : Units (WithZero (Multiplicative Int))
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LT.lt (HahnSeries.order f) 0
      η : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
      hη : Eq η (Units.mk0 ↑(Multiplicative.ofAdd (HahnSeries.order f)) ⋯)
      P : Polynomial K
      hP : LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑P)) ↑(HMul.hMul  …
      ⊢ LT.lt (Valued.v (HSub.hSub f ↑(HMul.hMul (HPow.hPow RatFunc.X (HahnSeries.or …
    -/
    have F_mul := f.ofPowerSeries_powerSeriesPart
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      f : LaurentSeries K
      γ : Units (WithZero (Multiplicative Int))
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LT.lt (HahnSeries.order f) 0
      η : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
      hη : Eq η (Units.mk0 ↑(Multiplicative.ofAdd (HahnSeries.order f)) ⋯)
      P : Polynomial K
      hP : LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑P)) ↑(HMul.hMul  …
      F_mul : Eq ((HahnSeries.ofPowerSeries Int K) f.powerSeriesPart) (HMul.hMul ((H …
      ⊢ LT.lt (Valued.v (HSub.hSub f ↑(HMul.hMul (HPow.hPow RatFunc.X (HahnSeries.or …
    -/
    obtain ⟨s, hs⟩ := Int.exists_eq_neg_ofNat (le_of_lt ord_nonpos)
    /-
      case h.intro
      K : Type u_2
      inst✝ : Field K
      f : LaurentSeries K
      γ : Units (WithZero (Multiplicative Int))
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : LT.lt (HahnSeries.order f) 0
      η : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
      hη : Eq η (Units.mk0 ↑(Multiplicative.ofAdd (HahnSeries.order f)) ⋯)
      P : Polynomial K
      hP : LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑P)) ↑(HMul.hMul  …
      F_mul : Eq ((HahnSeries.ofPowerSeries Int K) f.powerSeriesPart) (HMul.hMul ((H …
      s : Nat
      hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
      ⊢ LT.lt (Valued.v (HSub.hSub f ↑(HMul.hMul (HPow.hPow RatFunc.X (HahnSeries.or …
    -/
    rw [← hF, hs, neg_neg, ← ofPowerSeries_X_pow s, ← inv_mul_eq_iff_eq_mul₀] at F_mul
    · rw [hs, ← F_mul, PowerSeries.coe_pow, PowerSeries.coe_X, RatFunc.coe_mul, zpow_neg,
        zpow_natCast, inv_eq_one_div (RatFunc.X ^ s), RatFunc.coe_div, RatFunc.coe_pow,
        RatFunc.coe_X, RatFunc.coe_one, ← inv_eq_one_div, ← mul_sub, map_mul, map_inv₀,
        ← PowerSeries.coe_X, valuation_X_pow, ← hs, ← RatFunc.coe_coe, ← PowerSeries.coe_sub,
        ← coe_algebraMap, adicValued_apply, valuation_of_algebraMap,
        ← Units.val_mk0 (a := ((Multiplicative.ofAdd f.order : Multiplicative ℤ) : ℤₘ₀)), ← hη]
      /-
        case h.intro
        K : Type u_2
        inst✝ : Field K
        f : LaurentSeries K
        γ : Units (WithZero (Multiplicative Int))
        F : PowerSeries K := f.powerSeriesPart
        hF : Eq F f.powerSeriesPart
        ord_nonpos : LT.lt (HahnSeries.order f) 0
        η : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
        hη : Eq η (Units.mk0 ↑(Multiplicative.ofAdd (HahnSeries.order f)) ⋯)
        P : Polynomial K
        hP : LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑P)) ↑(HMul.hMul  …
        s : Nat
        F_mul : Eq (HMul.hMul (Inv.inv ((HahnSeries.ofPowerSeries Int K) (HPow.hPow Po …
        hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
        ⊢ LT.lt (HMul.hMul (Inv.inv ↑η) ((PowerSeries.idealX K).intValuation (HSub.hSu …
      -/
      apply inv_mul_lt_of_lt_mul₀
      /-
        case h.intro.h
        K : Type u_2
        inst✝ : Field K
        f : LaurentSeries K
        γ : Units (WithZero (Multiplicative Int))
        F : PowerSeries K := f.powerSeriesPart
        hF : Eq F f.powerSeriesPart
        ord_nonpos : LT.lt (HahnSeries.order f) 0
        η : Units (WithZero (Multiplicative Int)) := Units.mk0 ↑(Multiplicative.ofAdd  …
        hη : Eq η (Units.mk0 ↑(Multiplicative.ofAdd (HahnSeries.order f)) ⋯)
        P : Polynomial K
        hP : LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑P)) ↑(HMul.hMul  …
        s : Nat
        F_mul : Eq (HMul.hMul (Inv.inv ((HahnSeries.ofPowerSeries Int K) (HPow.hPow Po …
        hs : Eq (HahnSeries.order f) (Neg.neg ↑s)
        ⊢ LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub F ↑P)) (HMul.hMul ↑η ↑γ)
      -/
      rwa [← Units.val_mul]
      /-
        🎉 no goals
      -/
    · simp only [PowerSeries.coe_pow, pow_ne_zero, PowerSeries.coe_X, ne_eq,
        single_eq_zero_iff, one_ne_zero, not_false_iff]
    /-
      case neg
      K : Type u_2
      inst✝ : Field K
      f : LaurentSeries K
      γ : Units (WithZero (Multiplicative Int))
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : Not (LT.lt (HahnSeries.order f) 0)
      ⊢ Exists fun Q => LT.lt (Valued.v (HSub.hSub f ↑Q)) ↑γ
    -/
  · obtain ⟨s, hs⟩ := Int.exists_eq_neg_ofNat (Int.neg_nonpos_of_nonneg (not_lt.mp ord_nonpos))
    /-
      case neg.intro
      K : Type u_2
      inst✝ : Field K
      f : LaurentSeries K
      γ : Units (WithZero (Multiplicative Int))
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : Not (LT.lt (HahnSeries.order f) 0)
      s : Nat
      hs : Eq (Neg.neg (HahnSeries.order f)) (Neg.neg ↑s)
      ⊢ Exists fun Q => LT.lt (Valued.v (HSub.hSub f ↑Q)) ↑γ
    -/
    obtain ⟨P, hP⟩ := exists_Polynomial_intValuation_lt (PowerSeries.X ^ s * F) γ
    /-
      case neg.intro.intro
      K : Type u_2
      inst✝ : Field K
      f : LaurentSeries K
      γ : Units (WithZero (Multiplicative Int))
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : Not (LT.lt (HahnSeries.order f) 0)
      s : Nat
      hs : Eq (Neg.neg (HahnSeries.order f)) (Neg.neg ↑s)
      P : Polynomial K
      hP : LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub (HMul.hMul (HPow.hP …
      ⊢ Exists fun Q => LT.lt (Valued.v (HSub.hSub f ↑Q)) ↑γ
    -/
    use P
    rw [← X_order_mul_powerSeriesPart (neg_inj.1 hs).symm, ← RatFunc.coe_coe,
      ← PowerSeries.coe_sub, ← coe_algebraMap, adicValued_apply, valuation_of_algebraMap]
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      f : LaurentSeries K
      γ : Units (WithZero (Multiplicative Int))
      F : PowerSeries K := f.powerSeriesPart
      hF : Eq F f.powerSeriesPart
      ord_nonpos : Not (LT.lt (HahnSeries.order f) 0)
      s : Nat
      hs : Eq (Neg.neg (HahnSeries.order f)) (Neg.neg ↑s)
      P : Polynomial K
      hP : LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub (HMul.hMul (HPow.hP …
      ⊢ LT.lt ((PowerSeries.idealX K).intValuation (HSub.hSub (HMul.hMul (HPow.hPow  …
    -/
    exact hP
    /-
      🎉 no goals
    -/


theorem coe_range_dense : DenseRange ((↑) : RatFunc K → K⸨X⸩) := by
  /-
    K : Type u_2
    inst✝ : Field K
    ⊢ DenseRange RatFunc.coeToLaurentSeries_fun
  -/
  rw [denseRange_iff_closure_range]
  /-
    K : Type u_2
    inst✝ : Field K
    ⊢ Eq (closure (Set.range RatFunc.coeToLaurentSeries_fun)) Set.univ
  -/
  ext f
  simp only [UniformSpace.mem_closure_iff_symm_ball, Set.mem_univ, iff_true, Set.Nonempty,
    Set.mem_inter_iff, Set.mem_range, Set.mem_setOf_eq, exists_exists_eq_and]
  /-
    case h
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    ⊢ ∀ {V : Set (Prod (LaurentSeries K) (LaurentSeries K))}, Membership.mem (unif …
  -/
  intro V hV h_symm
  /-
    case h
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    V : Set (Prod (LaurentSeries K) (LaurentSeries K))
    hV : Membership.mem (uniformity (LaurentSeries K)) V
    h_symm : SymmetricRel V
    ⊢ Exists fun a => Membership.mem (UniformSpace.ball f V) ↑a
  -/
  rw [uniformity_eq_comap_neg_add_nhds_zero_swapped] at hV
  /-
    case h
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    V : Set (Prod (LaurentSeries K) (LaurentSeries K))
    hV : Membership.mem (Filter.comap (fun x => HAdd.hAdd (Neg.neg x.2) x.1) (nhds …
    h_symm : SymmetricRel V
    ⊢ Exists fun a => Membership.mem (UniformSpace.ball f V) ↑a
  -/
  obtain ⟨T, hT₀, hT₁⟩ := hV
  /-
    case h.intro.intro
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    V : Set (Prod (LaurentSeries K) (LaurentSeries K))
    h_symm : SymmetricRel V
    T : Set (LaurentSeries K)
    hT₀ : Membership.mem (nhds 0) T
    hT₁ : HasSubset.Subset (Set.preimage (fun x => HAdd.hAdd (Neg.neg x.2) x.1) T) V
    ⊢ Exists fun a => Membership.mem (UniformSpace.ball f V) ↑a
  -/
  obtain ⟨γ, hγ⟩ := Valued.mem_nhds_zero.mp hT₀
  /-
    case h.intro.intro.intro
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    V : Set (Prod (LaurentSeries K) (LaurentSeries K))
    h_symm : SymmetricRel V
    T : Set (LaurentSeries K)
    hT₀ : Membership.mem (nhds 0) T
    hT₁ : HasSubset.Subset (Set.preimage (fun x => HAdd.hAdd (Neg.neg x.2) x.1) T) V
    γ : Units (WithZero (Multiplicative Int))
    hγ : HasSubset.Subset (setOf fun x => LT.lt (Valued.v x) ↑γ) T
    ⊢ Exists fun a => Membership.mem (UniformSpace.ball f V) ↑a
  -/
  obtain ⟨P, _⟩ := exists_ratFunc_val_lt f γ
  /-
    case h.intro.intro.intro.intro
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    V : Set (Prod (LaurentSeries K) (LaurentSeries K))
    h_symm : SymmetricRel V
    T : Set (LaurentSeries K)
    hT₀ : Membership.mem (nhds 0) T
    hT₁ : HasSubset.Subset (Set.preimage (fun x => HAdd.hAdd (Neg.neg x.2) x.1) T) V
    γ : Units (WithZero (Multiplicative Int))
    hγ : HasSubset.Subset (setOf fun x => LT.lt (Valued.v x) ↑γ) T
    P : RatFunc K
    h✝ : LT.lt (Valued.v (HSub.hSub f ↑P)) ↑γ
    ⊢ Exists fun a => Membership.mem (UniformSpace.ball f V) ↑a
  -/
  use P
  /-
    case h
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    V : Set (Prod (LaurentSeries K) (LaurentSeries K))
    h_symm : SymmetricRel V
    T : Set (LaurentSeries K)
    hT₀ : Membership.mem (nhds 0) T
    hT₁ : HasSubset.Subset (Set.preimage (fun x => HAdd.hAdd (Neg.neg x.2) x.1) T) V
    γ : Units (WithZero (Multiplicative Int))
    hγ : HasSubset.Subset (setOf fun x => LT.lt (Valued.v x) ↑γ) T
    P : RatFunc K
    h✝ : LT.lt (Valued.v (HSub.hSub f ↑P)) ↑γ
    ⊢ Membership.mem (UniformSpace.ball f V) ↑P
  -/
  apply hT₁
  /-
    case h.a
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    V : Set (Prod (LaurentSeries K) (LaurentSeries K))
    h_symm : SymmetricRel V
    T : Set (LaurentSeries K)
    hT₀ : Membership.mem (nhds 0) T
    hT₁ : HasSubset.Subset (Set.preimage (fun x => HAdd.hAdd (Neg.neg x.2) x.1) T) V
    γ : Units (WithZero (Multiplicative Int))
    hγ : HasSubset.Subset (setOf fun x => LT.lt (Valued.v x) ↑γ) T
    P : RatFunc K
    h✝ : LT.lt (Valued.v (HSub.hSub f ↑P)) ↑γ
    ⊢ Membership.mem (Set.preimage (fun x => HAdd.hAdd (Neg.neg x.2) x.1) T) { fst …
  -/
  apply hγ
  /-
    case h.a.a
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    V : Set (Prod (LaurentSeries K) (LaurentSeries K))
    h_symm : SymmetricRel V
    T : Set (LaurentSeries K)
    hT₀ : Membership.mem (nhds 0) T
    hT₁ : HasSubset.Subset (Set.preimage (fun x => HAdd.hAdd (Neg.neg x.2) x.1) T) V
    γ : Units (WithZero (Multiplicative Int))
    hγ : HasSubset.Subset (setOf fun x => LT.lt (Valued.v x) ↑γ) T
    P : RatFunc K
    h✝ : LT.lt (Valued.v (HSub.hSub f ↑P)) ↑γ
    ⊢ Membership.mem (setOf fun x => LT.lt (Valued.v x) ↑γ) ((fun x => HAdd.hAdd ( …
  -/
  simpa only [add_comm, ← sub_eq_add_neg, gt_iff_lt, Set.mem_setOf_eq]
  /-
    🎉 no goals
  -/


theorem inducing_coe : IsUniformInducing ((↑) : RatFunc K → K⸨X⸩) := by
  /-
    K : Type u_2
    inst✝ : Field K
    ⊢ IsUniformInducing RatFunc.coeToLaurentSeries_fun
  -/
  rw [isUniformInducing_iff, Filter.comap]
  /-
    K : Type u_2
    inst✝ : Field K
    ⊢ Eq { sets := setOf fun s => Exists fun t => And (Membership.mem (uniformity  …
  -/
  ext S
  simp only [exists_prop, Filter.mem_mk, Set.mem_setOf_eq, uniformity_eq_comap_nhds_zero,
    Filter.mem_comap]
  /-
    case h
    K : Type u_2
    inst✝ : Field K
    S : Set (Prod (RatFunc K) (RatFunc K))
    ⊢ Iff (Exists fun t => And (Exists fun t_1 => And (Membership.mem (nhds 0) t_1 …
  -/
  constructor
    /-
      case h.mp
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      ⊢ (Exists fun t => And (Exists fun t_1 => And (Membership.mem (nhds 0) t_1) (H …
    -/
  · rintro ⟨T, ⟨⟨R, ⟨hR, pre_R⟩⟩, pre_T⟩⟩
    /-
      case h.mp.intro.intro.intro.intro
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      T : Set (Prod (LaurentSeries K) (LaurentSeries K))
      pre_T : HasSubset.Subset (Set.preimage (fun x => { fst := ↑x.1, snd := ↑x.2 }) …
      R : Set (LaurentSeries K)
      hR : Membership.mem (nhds 0) R
      pre_R : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) R) T
      ⊢ Exists fun t => And (Membership.mem (nhds 0) t) (HasSubset.Subset (Set.preim …
    -/
    obtain ⟨d, hd⟩ := Valued.mem_nhds.mp hR
    /-
      case h.mp.intro.intro.intro.intro.intro
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      T : Set (Prod (LaurentSeries K) (LaurentSeries K))
      pre_T : HasSubset.Subset (Set.preimage (fun x => { fst := ↑x.1, snd := ↑x.2 }) …
      R : Set (LaurentSeries K)
      hR : Membership.mem (nhds 0) R
      pre_R : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) R) T
      d : Units (WithZero (Multiplicative Int))
      hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) R
      ⊢ Exists fun t => And (Membership.mem (nhds 0) t) (HasSubset.Subset (Set.preim …
    -/
    use {P : RatFunc K | Valued.v P < ↑d}
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      T : Set (Prod (LaurentSeries K) (LaurentSeries K))
      pre_T : HasSubset.Subset (Set.preimage (fun x => { fst := ↑x.1, snd := ↑x.2 }) …
      R : Set (LaurentSeries K)
      hR : Membership.mem (nhds 0) R
      pre_R : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) R) T
      d : Units (WithZero (Multiplicative Int))
      hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) R
      ⊢ And (Membership.mem (nhds 0) (setOf fun P => LT.lt (Valued.v P) ↑d)) (HasSub …
    -/
    simp only [Valued.mem_nhds, sub_zero]
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      T : Set (Prod (LaurentSeries K) (LaurentSeries K))
      pre_T : HasSubset.Subset (Set.preimage (fun x => { fst := ↑x.1, snd := ↑x.2 }) …
      R : Set (LaurentSeries K)
      hR : Membership.mem (nhds 0) R
      pre_R : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) R) T
      d : Units (WithZero (Multiplicative Int))
      hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) R
      ⊢ And (Exists fun γ => HasSubset.Subset (setOf fun y => LT.lt (Valued.v y) ↑γ) …
    -/
    refine ⟨⟨d, by rfl⟩, subset_trans (fun _ _ ↦ pre_R ?_) pre_T⟩
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      T : Set (Prod (LaurentSeries K) (LaurentSeries K))
      pre_T : HasSubset.Subset (Set.preimage (fun x => { fst := ↑x.1, snd := ↑x.2 }) …
      R : Set (LaurentSeries K)
      hR : Membership.mem (nhds 0) R
      pre_R : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) R) T
      d : Units (WithZero (Multiplicative Int))
      hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) R
      x✝¹ : Prod (RatFunc K) (RatFunc K)
      x✝ : Membership.mem (Set.preimage (fun x => HSub.hSub x.2 x.1) (setOf fun P => …
      ⊢ Membership.mem (Set.preimage (fun x => HSub.hSub x.2 x.1) R) { fst := ↑x✝¹.1 …
    -/
    apply hd
    /-
      case h.a
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      T : Set (Prod (LaurentSeries K) (LaurentSeries K))
      pre_T : HasSubset.Subset (Set.preimage (fun x => { fst := ↑x.1, snd := ↑x.2 }) …
      R : Set (LaurentSeries K)
      hR : Membership.mem (nhds 0) R
      pre_R : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) R) T
      d : Units (WithZero (Multiplicative Int))
      hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) R
      x✝¹ : Prod (RatFunc K) (RatFunc K)
      x✝ : Membership.mem (Set.preimage (fun x => HSub.hSub x.2 x.1) (setOf fun P => …
      ⊢ Membership.mem (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) ((fun x  …
    -/
    simp only [sub_zero, Set.mem_setOf_eq]
    /-
      case h.a
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      T : Set (Prod (LaurentSeries K) (LaurentSeries K))
      pre_T : HasSubset.Subset (Set.preimage (fun x => { fst := ↑x.1, snd := ↑x.2 }) …
      R : Set (LaurentSeries K)
      hR : Membership.mem (nhds 0) R
      pre_R : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) R) T
      d : Units (WithZero (Multiplicative Int))
      hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) R
      x✝¹ : Prod (RatFunc K) (RatFunc K)
      x✝ : Membership.mem (Set.preimage (fun x => HSub.hSub x.2 x.1) (setOf fun P => …
      ⊢ LT.lt (Valued.v (HSub.hSub ↑x✝¹.2 ↑x✝¹.1)) ↑d
    -/
    erw [← RatFunc.coe_sub, ← valuation_eq_LaurentSeries_valuation]
    /-
      case h.a
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      T : Set (Prod (LaurentSeries K) (LaurentSeries K))
      pre_T : HasSubset.Subset (Set.preimage (fun x => { fst := ↑x.1, snd := ↑x.2 }) …
      R : Set (LaurentSeries K)
      hR : Membership.mem (nhds 0) R
      pre_R : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) R) T
      d : Units (WithZero (Multiplicative Int))
      hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) R
      x✝¹ : Prod (RatFunc K) (RatFunc K)
      x✝ : Membership.mem (Set.preimage (fun x => HSub.hSub x.2 x.1) (setOf fun P => …
      ⊢ LT.lt ((Polynomial.idealX K).valuation (HSub.hSub x✝¹.2 x✝¹.1)) ↑d
    -/
    assumption
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      ⊢ (Exists fun t => And (Membership.mem (nhds 0) t) (HasSubset.Subset (Set.prei …
    -/
  · rintro ⟨_, ⟨hT, pre_T⟩⟩
    /-
      case h.mpr.intro.intro
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      w✝ : Set (RatFunc K)
      hT : Membership.mem (nhds 0) w✝
      pre_T : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) w✝) S
      ⊢ Exists fun t => And (Exists fun t_1 => And (Membership.mem (nhds 0) t_1) (Ha …
    -/
    obtain ⟨d, hd⟩ := Valued.mem_nhds.mp hT
    /-
      case h.mpr.intro.intro.intro
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      w✝ : Set (RatFunc K)
      hT : Membership.mem (nhds 0) w✝
      pre_T : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) w✝) S
      d : Units (WithZero (Multiplicative Int))
      hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) w✝
      ⊢ Exists fun t => And (Exists fun t_1 => And (Membership.mem (nhds 0) t_1) (Ha …
    -/
    let X := {f : K⸨X⸩ | Valued.v f < ↑d}
    /-
      case h.mpr.intro.intro.intro
      K : Type u_2
      inst✝ : Field K
      S : Set (Prod (RatFunc K) (RatFunc K))
      w✝ : Set (RatFunc K)
      hT : Membership.mem (nhds 0) w✝
      pre_T : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) w✝) S
      d : Units (WithZero (Multiplicative Int))
      hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) w✝
      X : Set (LaurentSeries K) := setOf fun f => LT.lt (Valued.v f) ↑d
      ⊢ Exists fun t => And (Exists fun t_1 => And (Membership.mem (nhds 0) t_1) (Ha …
    -/
    refine ⟨(fun x : K⸨X⸩ × K⸨X⸩ ↦ x.snd - x.fst) ⁻¹' X, ⟨X, ?_⟩, ?_⟩
      /-
        case h.mpr.intro.intro.intro.refine_1
        K : Type u_2
        inst✝ : Field K
        S : Set (Prod (RatFunc K) (RatFunc K))
        w✝ : Set (RatFunc K)
        hT : Membership.mem (nhds 0) w✝
        pre_T : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) w✝) S
        d : Units (WithZero (Multiplicative Int))
        hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) w✝
        X : Set (LaurentSeries K) := setOf fun f => LT.lt (Valued.v f) ↑d
        ⊢ And (Membership.mem (nhds 0) X) (HasSubset.Subset (Set.preimage (fun x => HS …
      -/
    · refine ⟨?_, Set.Subset.refl _⟩
        /-
          case h.mpr.intro.intro.intro.refine_1
          K : Type u_2
          inst✝ : Field K
          S : Set (Prod (RatFunc K) (RatFunc K))
          w✝ : Set (RatFunc K)
          hT : Membership.mem (nhds 0) w✝
          pre_T : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) w✝) S
          d : Units (WithZero (Multiplicative Int))
          hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) w✝
          X : Set (LaurentSeries K) := setOf fun f => LT.lt (Valued.v f) ↑d
          ⊢ Membership.mem (nhds 0) X
        -/
      · simp only [Valued.mem_nhds, sub_zero]
        /-
          case h.mpr.intro.intro.intro.refine_1
          K : Type u_2
          inst✝ : Field K
          S : Set (Prod (RatFunc K) (RatFunc K))
          w✝ : Set (RatFunc K)
          hT : Membership.mem (nhds 0) w✝
          pre_T : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) w✝) S
          d : Units (WithZero (Multiplicative Int))
          hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) w✝
          X : Set (LaurentSeries K) := setOf fun f => LT.lt (Valued.v f) ↑d
          ⊢ Exists fun γ => HasSubset.Subset (setOf fun y => LT.lt (Valued.v y) ↑γ) X
        -/
        use d
        /-
          🎉 no goals
        -/
      /-
        case h.mpr.intro.intro.intro.refine_2
        K : Type u_2
        inst✝ : Field K
        S : Set (Prod (RatFunc K) (RatFunc K))
        w✝ : Set (RatFunc K)
        hT : Membership.mem (nhds 0) w✝
        pre_T : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) w✝) S
        d : Units (WithZero (Multiplicative Int))
        hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) w✝
        X : Set (LaurentSeries K) := setOf fun f => LT.lt (Valued.v f) ↑d
        ⊢ HasSubset.Subset (Set.preimage (fun x => { fst := ↑x.1, snd := ↑x.2 }) (Set. …
      -/
    · refine subset_trans (fun _ _ ↦ ?_) pre_T
      /-
        case h.mpr.intro.intro.intro.refine_2
        K : Type u_2
        inst✝ : Field K
        S : Set (Prod (RatFunc K) (RatFunc K))
        w✝ : Set (RatFunc K)
        hT : Membership.mem (nhds 0) w✝
        pre_T : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) w✝) S
        d : Units (WithZero (Multiplicative Int))
        hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) w✝
        X : Set (LaurentSeries K) := setOf fun f => LT.lt (Valued.v f) ↑d
        x✝¹ : Prod (RatFunc K) (RatFunc K)
        x✝ : Membership.mem (Set.preimage (fun x => { fst := ↑x.1, snd := ↑x.2 }) (Set …
        ⊢ Membership.mem (Set.preimage (fun x => HSub.hSub x.2 x.1) w✝) x✝¹
      -/
      apply hd
      erw [Set.mem_setOf_eq, sub_zero, valuation_eq_LaurentSeries_valuation,
        RatFunc.coe_sub]
      /-
        case h.mpr.intro.intro.intro.refine_2.a
        K : Type u_2
        inst✝ : Field K
        S : Set (Prod (RatFunc K) (RatFunc K))
        w✝ : Set (RatFunc K)
        hT : Membership.mem (nhds 0) w✝
        pre_T : HasSubset.Subset (Set.preimage (fun x => HSub.hSub x.2 x.1) w✝) S
        d : Units (WithZero (Multiplicative Int))
        hd : HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y 0)) ↑d) w✝
        X : Set (LaurentSeries K) := setOf fun f => LT.lt (Valued.v f) ↑d
        x✝¹ : Prod (RatFunc K) (RatFunc K)
        x✝ : Membership.mem (Set.preimage (fun x => { fst := ↑x.1, snd := ↑x.2 }) (Set …
        ⊢ LT.lt ((PowerSeries.idealX K).valuation (HSub.hSub ↑x✝¹.2 ↑x✝¹.1)) ↑d
      -/
      assumption
      /-
        🎉 no goals
      -/


theorem continuous_coe : Continuous ((↑) : RatFunc K → K⸨X⸩) :=
  (isUniformInducing_iff'.1 (inducing_coe)).1.continuous


/-- The `X`-adic completion as an abstract completion of `RatFunc K`-/
abbrev ratfuncAdicComplPkg : AbstractCompletion (RatFunc K) :=
  UniformSpace.Completion.cPkg


/-- Having established that the `K⸨X⸩` is complete and contains `RatFunc K` as a dense
subspace, it gives rise to an abstract completion of `RatFunc K`.-/
noncomputable def LaurentSeriesPkg : AbstractCompletion (RatFunc K) where
  space := K⸨X⸩
  coe := (↑)
  uniformStruct := inferInstance
  complete := inferInstance
  separation := inferInstance
  isUniformInducing := inducing_coe
  dense := coe_range_dense


instance : TopologicalSpace (LaurentSeriesPkg K).space :=
  (LaurentSeriesPkg K).uniformStruct.toTopologicalSpace


@[simp]
theorem LaurentSeries_coe (x : RatFunc K) : (LaurentSeriesPkg K).coe x = (x : K⸨X⸩) :=
  rfl


/-- Reintrerpret the extension of `coe : RatFunc K → K⸨X⸩` as ring homomorphism -/
abbrev extensionAsRingHom :=
  UniformSpace.Completion.extensionHom (coeAlgHom K).toRingHom


/-- An abbreviation for the `X`-adic completion of `RatFunc K` -/
abbrev RatFuncAdicCompl := adicCompletion (RatFunc K) (idealX K)

/- The two instances below make `comparePkg` and `comparePkg_eq_extension` slightly faster-/

instance : UniformSpace (RatFuncAdicCompl K) := inferInstance

instance : UniformSpace K⸨X⸩ := inferInstance


/-- The uniform space isomorphism between two abstract completions of `ratfunc K` -/
abbrev comparePkg : RatFuncAdicCompl K ≃ᵤ K⸨X⸩ :=
  compareEquiv ratfuncAdicComplPkg (LaurentSeriesPkg K)


lemma comparePkg_eq_extension (x : UniformSpace.Completion (RatFunc K)) :
    (comparePkg K).toFun x = (extensionAsRingHom K (continuous_coe)).toFun x := rfl


/-- The uniform space equivalence between two abstract completions of `ratfunc K` as a ring
equivalence: this will be the *inverse* of the fundamental one.-/
abbrev ratfuncAdicComplRingEquiv : RatFuncAdicCompl K ≃+* K⸨X⸩ :=
  {comparePkg K with
    map_mul' := by
      /-
        R : Type u_1
        K : Type u_2
        inst✝ : Field K
        ⊢ ∀ (x y : LaurentSeries.RatFuncAdicCompl K), Eq (__src✝.toFun (HMul.hMul x y) …
      -/
      intro x y
      /-
        R : Type u_1
        K : Type u_2
        inst✝ : Field K
        x y : LaurentSeries.RatFuncAdicCompl K
        ⊢ Eq (__src✝.toFun (HMul.hMul x y)) (HMul.hMul (__src✝.toFun x) (__src✝.toFun  …
      -/
      rw [comparePkg_eq_extension, (extensionAsRingHom K (continuous_coe)).map_mul']
      /-
        R : Type u_1
        K : Type u_2
        inst✝ : Field K
        x y : LaurentSeries.RatFuncAdicCompl K
        ⊢ Eq (HMul.hMul ((↑↑(LaurentSeries.extensionAsRingHom K ⋯)).toFun x) ((↑↑(Laur …
      -/
      rfl
      /-
        🎉 no goals
      -/
    map_add' := by
      /-
        R : Type u_1
        K : Type u_2
        inst✝ : Field K
        ⊢ ∀ (x y : LaurentSeries.RatFuncAdicCompl K), Eq (__src✝.toFun (HAdd.hAdd x y) …
      -/
      intro x y
      /-
        R : Type u_1
        K : Type u_2
        inst✝ : Field K
        x y : LaurentSeries.RatFuncAdicCompl K
        ⊢ Eq (__src✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd (__src✝.toFun x) (__src✝.toFun  …
      -/
      rw [comparePkg_eq_extension, (extensionAsRingHom K (continuous_coe)).map_add']
      /-
        R : Type u_1
        K : Type u_2
        inst✝ : Field K
        x y : LaurentSeries.RatFuncAdicCompl K
        ⊢ Eq (HAdd.hAdd ((↑↑(LaurentSeries.extensionAsRingHom K ⋯)).toFun x) ((↑↑(Laur …
      -/
      rfl }
      /-
        🎉 no goals
      -/


/-- The uniform space equivalence between two abstract completions of `ratfunc K` as a ring
equivalence: it goes from `K⸨X⸩` to `RatFuncAdicCompl K` -/
abbrev LaurentSeriesRingEquiv : K⸨X⸩ ≃+* RatFuncAdicCompl K :=
  (ratfuncAdicComplRingEquiv K).symm


@[simp]
theorem ratfuncAdicComplRingEquiv_apply (x : RatFuncAdicCompl K) :
    ratfuncAdicComplRingEquiv K x = ratfuncAdicComplPkg.compare (LaurentSeriesPkg K) x := rfl


theorem coe_X_compare :
    (ratfuncAdicComplRingEquiv K) ((RatFunc.X : RatFunc K) : RatFuncAdicCompl K) =
      ((PowerSeries.X : K⟦X⟧) : K⸨X⸩) := by
  /-
    K : Type u_2
    inst✝ : Field K
    ⊢ Eq ((LaurentSeries.ratfuncAdicComplRingEquiv K) (↑(RatFunc K) RatFunc.X)) (( …
  -/
  rw [PowerSeries.coe_X, ← RatFunc.coe_X, ← LaurentSeries_coe, ← compare_coe]
  /-
    K : Type u_2
    inst✝ : Field K
    ⊢ Eq ((LaurentSeries.ratfuncAdicComplRingEquiv K) (↑(RatFunc K) RatFunc.X)) (A …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem valuation_LaurentSeries_equal_extension :
    (LaurentSeriesPkg K).isDenseInducing.extend Valued.v = (Valued.v : K⸨X⸩ → ℤₘ₀) := by
  /-
    K : Type u_2
    inst✝ : Field K
    ⊢ Eq (⋯.extend ⇑Valued.v) ⇑Valued.v
  -/
  apply IsDenseInducing.extend_unique
    /-
      case hf
      K : Type u_2
      inst✝ : Field K
      ⊢ ∀ (x : RatFunc K), Eq (Valued.v ((LaurentSeries.LaurentSeriesPkg K).coe x))  …
    -/
  · intro x
    /-
      case hf
      K : Type u_2
      inst✝ : Field K
      x : RatFunc K
      ⊢ Eq (Valued.v ((LaurentSeries.LaurentSeriesPkg K).coe x)) (Valued.v x)
    -/
    erw [valuation_eq_LaurentSeries_valuation K x]
    /-
      case hf
      K : Type u_2
      inst✝ : Field K
      x : RatFunc K
      ⊢ Eq (Valued.v ((LaurentSeries.LaurentSeriesPkg K).coe x)) ((PowerSeries.ideal …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case hg
      K : Type u_2
      inst✝ : Field K
      ⊢ Continuous ⇑Valued.v
    -/
  · exact Valued.continuous_valuation (K := K⸨X⸩)
    /-
      🎉 no goals
    -/


theorem tendsto_valuation (a : (idealX K).adicCompletion (RatFunc K)) :
    Tendsto (Valued.v : RatFunc K → ℤₘ₀) (comap (↑) (𝓝 a)) (𝓝 (Valued.v a : ℤₘ₀)) := by
  /-
    K : Type u_2
    inst✝ : Field K
    a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
    ⊢ Filter.Tendsto (⇑Valued.v) (Filter.comap (↑(RatFunc K)) (nhds a)) (nhds (Val …
  -/
  set ψ := (Valued.v : RatFunc K → ℤₘ₀) with hψ
  /-
    K : Type u_2
    inst✝ : Field K
    a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
    ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
    hψ : Eq ψ ⇑Valued.v
    ⊢ Filter.Tendsto ψ (Filter.comap (↑(RatFunc K)) (nhds a)) (nhds (Valued.v a))
  -/
  have := Valued.is_topological_valuation (R := (idealX K).adicCompletion (RatFunc K))
  /-
    K : Type u_2
    inst✝ : Field K
    a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
    ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
    hψ : Eq ψ ⇑Valued.v
    this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
    ⊢ Filter.Tendsto ψ (Filter.comap (↑(RatFunc K)) (nhds a)) (nhds (Valued.v a))
  -/
  by_cases ha : a = 0
    /-
      case pos
      K : Type u_2
      inst✝ : Field K
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
      hψ : Eq ψ ⇑Valued.v
      this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
      ha : Eq a 0
      ⊢ Filter.Tendsto ψ (Filter.comap (↑(RatFunc K)) (nhds a)) (nhds (Valued.v a))
    -/
  · rw [tendsto_def]
    /-
      case pos
      K : Type u_2
      inst✝ : Field K
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
      hψ : Eq ψ ⇑Valued.v
      this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
      ha : Eq a 0
      ⊢ ∀ (s : Set (WithZero (Multiplicative Int))), Membership.mem (nhds (Valued.v  …
    -/
    intro S hS
    /-
      case pos
      K : Type u_2
      inst✝ : Field K
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
      hψ : Eq ψ ⇑Valued.v
      this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
      ha : Eq a 0
      S : Set (WithZero (Multiplicative Int))
      hS : Membership.mem (nhds (Valued.v a)) S
      ⊢ Membership.mem (Filter.comap (↑(RatFunc K)) (nhds a)) (Set.preimage ψ S)
    -/
    rw [ha, map_zero, WithZeroTopology.hasBasis_nhds_zero.1 S] at hS
    /-
      case pos
      K : Type u_2
      inst✝ : Field K
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
      hψ : Eq ψ ⇑Valued.v
      this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
      ha : Eq a 0
      S : Set (WithZero (Multiplicative Int))
      hS : Exists fun i => And (Ne i 0) (HasSubset.Subset (Set.Iio i) S)
      ⊢ Membership.mem (Filter.comap (↑(RatFunc K)) (nhds a)) (Set.preimage ψ S)
    -/
    obtain ⟨γ, γ_ne_zero, γ_le⟩ := hS
    /-
      case pos.intro.intro
      K : Type u_2
      inst✝ : Field K
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
      hψ : Eq ψ ⇑Valued.v
      this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
      ha : Eq a 0
      S : Set (WithZero (Multiplicative Int))
      γ : WithZero (Multiplicative Int)
      γ_ne_zero : Ne γ 0
      γ_le : HasSubset.Subset (Set.Iio γ) S
      ⊢ Membership.mem (Filter.comap (↑(RatFunc K)) (nhds a)) (Set.preimage ψ S)
    -/
    use {t | Valued.v t < γ}
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
      hψ : Eq ψ ⇑Valued.v
      this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
      ha : Eq a 0
      S : Set (WithZero (Multiplicative Int))
      γ : WithZero (Multiplicative Int)
      γ_ne_zero : Ne γ 0
      γ_le : HasSubset.Subset (Set.Iio γ) S
      ⊢ And (Membership.mem (nhds a) (setOf fun t => LT.lt (Valued.v t) γ)) (HasSubs …
    -/
    constructor
      /-
        case h.left
        K : Type u_2
        inst✝ : Field K
        a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
        ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
        hψ : Eq ψ ⇑Valued.v
        this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
        ha : Eq a 0
        S : Set (WithZero (Multiplicative Int))
        γ : WithZero (Multiplicative Int)
        γ_ne_zero : Ne γ 0
        γ_le : HasSubset.Subset (Set.Iio γ) S
        ⊢ Membership.mem (nhds a) (setOf fun t => LT.lt (Valued.v t) γ)
      -/
    · rw [ha, this]
      /-
        case h.left
        K : Type u_2
        inst✝ : Field K
        a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
        ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
        hψ : Eq ψ ⇑Valued.v
        this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
        ha : Eq a 0
        S : Set (WithZero (Multiplicative Int))
        γ : WithZero (Multiplicative Int)
        γ_ne_zero : Ne γ 0
        γ_le : HasSubset.Subset (Set.Iio γ) S
        ⊢ Exists fun γ_1 => HasSubset.Subset (setOf fun x => LT.lt (Valued.v x) ↑γ_1)  …
      -/
      use Units.mk0 γ γ_ne_zero
      /-
        case h
        K : Type u_2
        inst✝ : Field K
        a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
        ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
        hψ : Eq ψ ⇑Valued.v
        this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
        ha : Eq a 0
        S : Set (WithZero (Multiplicative Int))
        γ : WithZero (Multiplicative Int)
        γ_ne_zero : Ne γ 0
        γ_le : HasSubset.Subset (Set.Iio γ) S
        ⊢ HasSubset.Subset (setOf fun x => LT.lt (Valued.v x) ↑(Units.mk0 γ γ_ne_zero) …
      -/
      rw [Units.val_mk0]
      /-
        🎉 no goals
      -/
      /-
        case h.right
        K : Type u_2
        inst✝ : Field K
        a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
        ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
        hψ : Eq ψ ⇑Valued.v
        this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
        ha : Eq a 0
        S : Set (WithZero (Multiplicative Int))
        γ : WithZero (Multiplicative Int)
        γ_ne_zero : Ne γ 0
        γ_le : HasSubset.Subset (Set.Iio γ) S
        ⊢ HasSubset.Subset (Set.preimage (↑(RatFunc K)) (setOf fun t => LT.lt (Valued. …
      -/
    · refine Set.Subset.trans (fun a _ ↦ ?_) (Set.preimage_mono γ_le)
      /-
        case h.right
        K : Type u_2
        inst✝ : Field K
        a✝ : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial …
        ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
        hψ : Eq ψ ⇑Valued.v
        this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
        ha : Eq a✝ 0
        S : Set (WithZero (Multiplicative Int))
        γ : WithZero (Multiplicative Int)
        γ_ne_zero : Ne γ 0
        γ_le : HasSubset.Subset (Set.Iio γ) S
        a : RatFunc K
        x✝ : Membership.mem (Set.preimage (↑(RatFunc K)) (setOf fun t => LT.lt (Valued …
        ⊢ Membership.mem (Set.preimage ψ (Set.Iio γ)) a
      -/
      rwa [Set.mem_preimage, Set.mem_Iio, hψ, ← Valued.valuedCompletion_apply a]
      /-
        🎉 no goals
      -/
  · rw [WithZeroTopology.tendsto_of_ne_zero ((Valuation.ne_zero_iff Valued.v).mpr ha), hψ,
      Filter.eventually_comap, Filter.Eventually, Valued.mem_nhds]
    /-
      case neg
      K : Type u_2
      inst✝ : Field K
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
      hψ : Eq ψ ⇑Valued.v
      this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
      ha : Not (Eq a 0)
      ⊢ Exists fun γ => HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub  …
    -/
    set γ := Valued.v a / (↑(Multiplicative.ofAdd (1 : ℤ)) : ℤₘ₀) with h_aγ
    have γ_ne_zero : γ ≠ 0 := by
      rw [ne_eq, _root_.div_eq_zero_iff, Valuation.zero_iff]
      simpa only [coe_ne_zero, or_false]
    /-
      case neg
      K : Type u_2
      inst✝ : Field K
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
      hψ : Eq ψ ⇑Valued.v
      this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
      ha : Not (Eq a 0)
      γ : WithZero (Multiplicative Int) := HDiv.hDiv (Valued.v a) ↑(Multiplicative.o …
      h_aγ : Eq γ (HDiv.hDiv (Valued.v a) ↑(Multiplicative.ofAdd 1))
      γ_ne_zero : Ne γ 0
      ⊢ Exists fun γ => HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub  …
    -/
    use Units.mk0 γ γ_ne_zero
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
      hψ : Eq ψ ⇑Valued.v
      this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
      ha : Not (Eq a 0)
      γ : WithZero (Multiplicative Int) := HDiv.hDiv (Valued.v a) ↑(Multiplicative.o …
      h_aγ : Eq γ (HDiv.hDiv (Valued.v a) ↑(Multiplicative.ofAdd 1))
      γ_ne_zero : Ne γ 0
      ⊢ HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y a)) ↑(Units.mk …
    -/
    intro y val_y b diff_b_y
    replace val_y : Valued.v y = Valued.v a := by
      refine Valuation.map_eq_of_sub_lt _ (val_y.trans ?_)
      rw [Units.val_mk0, h_aγ, ← coe_unzero ((Valuation.ne_zero_iff Valued.v).mpr ha), ←
        WithZero.coe_div, coe_lt_coe, div_lt_self_iff, ← ofAdd_zero,
        Multiplicative.ofAdd_lt]
      exact Int.zero_lt_one
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
      hψ : Eq ψ ⇑Valued.v
      this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
      ha : Not (Eq a 0)
      γ : WithZero (Multiplicative Int) := HDiv.hDiv (Valued.v a) ↑(Multiplicative.o …
      h_aγ : Eq γ (HDiv.hDiv (Valued.v a) ↑(Multiplicative.ofAdd 1))
      γ_ne_zero : Ne γ 0
      y : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      b : RatFunc K
      diff_b_y : Eq (↑(RatFunc K) b) y
      val_y : Eq (Valued.v y) (Valued.v a)
      ⊢ Eq (Valued.v b) (Valued.v a)
    -/
    rw [← Valued.extension_extends, ← val_y, ← diff_b_y]
    /-
      case h
      K : Type u_2
      inst✝ : Field K
      a : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      ψ : RatFunc K → WithZero (Multiplicative Int) := ⇑Valued.v
      hψ : Eq ψ ⇑Valued.v
      this : ∀ (s : Set (IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc  …
      ha : Not (Eq a 0)
      γ : WithZero (Multiplicative Int) := HDiv.hDiv (Valued.v a) ↑(Multiplicative.o …
      h_aγ : Eq γ (HDiv.hDiv (Valued.v a) ↑(Multiplicative.ofAdd 1))
      γ_ne_zero : Ne γ 0
      y : IsDedekindDomain.HeightOneSpectrum.adicCompletion (RatFunc K) (Polynomial. …
      b : RatFunc K
      diff_b_y : Eq (↑(RatFunc K) b) y
      val_y : Eq (Valued.v y) (Valued.v a)
      ⊢ Eq (Valued.extension (↑(RatFunc K) b)) (Valued.v (↑(RatFunc K) b))
    -/
    congr
    /-
      🎉 no goals
    -/

/- The extension of the `X`-adic valuation from `RatFunc K` up to its abstract completion coincides,
modulo the isomorphism with `K⸨X⸩`, with the `X`-adic valuation on `K⸨X⸩`. -/

theorem valuation_compare (f : K⸨X⸩) :
    (Valued.v : (RatFuncAdicCompl K) → ℤₘ₀)
        (AbstractCompletion.compare (LaurentSeriesPkg K) ratfuncAdicComplPkg f) =
      Valued.v f := by
  rw [← valuation_LaurentSeries_equal_extension, ← compare_comp_eq_compare
    (pkg := ratfuncAdicComplPkg) (cont_f := Valued.continuous_valuation)]
    /-
      K : Type u_2
      inst✝ : Field K
      f : LaurentSeries K
      ⊢ Eq (Valued.v ((LaurentSeries.LaurentSeriesPkg K).compare LaurentSeries.ratfu …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case a
    K : Type u_2
    inst✝ : Field K
    f : LaurentSeries K
    ⊢ ∀ (a : LaurentSeries.ratfuncAdicComplPkg.space), Filter.Tendsto (⇑Valued.v)  …
  -/
  exact (tendsto_valuation K)
  /-
    🎉 no goals
  -/


/-- In order to compare `K⟦X⟧` with the valuation subring in the `X`-adic completion of
`RatFunc K` we consider its alias as a subring of `K⸨X⸩`. -/
abbrev powerSeries_as_subring : Subring K⸨X⸩ :=
  RingHom.range (HahnSeries.ofPowerSeries ℤ K)


/-- The ring `K⟦X⟧` is isomorphic to the subring `powerSeries_as_subring K` -/
abbrev powerSeriesEquivSubring : K⟦X⟧ ≃+* powerSeries_as_subring K := by
  /-
    R : Type u_1
    K : Type u_2
    inst✝ : Field K
    ⊢ RingEquiv (PowerSeries K) (Subtype fun x => Membership.mem (LaurentSeries.po …
  -/
  rw [powerSeries_as_subring, RingHom.range_eq_map]
  exact ((Subring.topEquiv).symm).trans (Subring.equivMapOfInjective ⊤ (ofPowerSeries ℤ K)
    ofPowerSeries_injective)

/- Through the isomorphism `LaurentSeriesRingEquiv`, power series land in the unit ball inside the
completion of `RatFunc K`. -/

theorem mem_integers_of_powerSeries (F : K⟦X⟧) :
    (LaurentSeriesRingEquiv K) F ∈ (idealX K).adicCompletionIntegers (RatFunc K) := by
  have : (LaurentSeriesRingEquiv K) F =
    (LaurentSeriesPkg K).compare ratfuncAdicComplPkg (F : K⸨X⸩) := rfl
  simp only [Subring.mem_map, exists_prop, ValuationSubring.mem_toSubring,
    mem_adicCompletionIntegers, this,  valuation_compare, val_le_one_iff_eq_coe]
  /-
    K : Type u_2
    inst✝ : Field K
    F : PowerSeries K
    this : Eq ((LaurentSeries.LaurentSeriesRingEquiv K) ((HahnSeries.ofPowerSeries …
    ⊢ Exists fun F_1 => Eq ((HahnSeries.ofPowerSeries Int K) F_1) ((HahnSeries.ofP …
  -/
  exact ⟨F, rfl⟩
  /-
    🎉 no goals
  -/

/- Conversely, all elements in the unit ball inside the completion of `RatFunc K` come from a power
series through the isomorphism `LaurentSeriesRingEquiv`. -/

theorem exists_powerSeries_of_memIntegers {x : RatFuncAdicCompl K}
    (hx : x ∈ (idealX K).adicCompletionIntegers (RatFunc K)) :
    ∃ F : K⟦X⟧, (LaurentSeriesRingEquiv K) F = x := by
  /-
    K : Type u_2
    inst✝ : Field K
    x : LaurentSeries.RatFuncAdicCompl K
    hx : Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers …
    ⊢ Exists fun F => Eq ((LaurentSeries.LaurentSeriesRingEquiv K) ((HahnSeries.of …
  -/
  set f := (ratfuncAdicComplRingEquiv K) x with hf
  have H_x : (LaurentSeriesPkg K).compare ratfuncAdicComplPkg ((ratfuncAdicComplRingEquiv K) x) =
      x := congr_fun (inverse_compare (LaurentSeriesPkg K) ratfuncAdicComplPkg) x
  /-
    K : Type u_2
    inst✝ : Field K
    x : LaurentSeries.RatFuncAdicCompl K
    hx : Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers …
    f : LaurentSeries K := (LaurentSeries.ratfuncAdicComplRingEquiv K) x
    hf : Eq f ((LaurentSeries.ratfuncAdicComplRingEquiv K) x)
    H_x : Eq ((LaurentSeries.LaurentSeriesPkg K).compare LaurentSeries.ratfuncAdic …
    ⊢ Exists fun F => Eq ((LaurentSeries.LaurentSeriesRingEquiv K) ((HahnSeries.of …
  -/
  rw [mem_adicCompletionIntegers, ← H_x] at hx
  /-
    K : Type u_2
    inst✝ : Field K
    x : LaurentSeries.RatFuncAdicCompl K
    hx : LE.le (Valued.v ((LaurentSeries.LaurentSeriesPkg K).compare LaurentSeries …
    f : LaurentSeries K := (LaurentSeries.ratfuncAdicComplRingEquiv K) x
    hf : Eq f ((LaurentSeries.ratfuncAdicComplRingEquiv K) x)
    H_x : Eq ((LaurentSeries.LaurentSeriesPkg K).compare LaurentSeries.ratfuncAdic …
    ⊢ Exists fun F => Eq ((LaurentSeries.LaurentSeriesRingEquiv K) ((HahnSeries.of …
  -/
  obtain ⟨F, hF⟩ := (val_le_one_iff_eq_coe K f).mp (valuation_compare _ f ▸ hx)
  /-
    case intro
    K : Type u_2
    inst✝ : Field K
    x : LaurentSeries.RatFuncAdicCompl K
    hx : LE.le (Valued.v ((LaurentSeries.LaurentSeriesPkg K).compare LaurentSeries …
    f : LaurentSeries K := (LaurentSeries.ratfuncAdicComplRingEquiv K) x
    hf : Eq f ((LaurentSeries.ratfuncAdicComplRingEquiv K) x)
    H_x : Eq ((LaurentSeries.LaurentSeriesPkg K).compare LaurentSeries.ratfuncAdic …
    F : PowerSeries K
    hF : Eq ((HahnSeries.ofPowerSeries Int K) F) f
    ⊢ Exists fun F => Eq ((LaurentSeries.LaurentSeriesRingEquiv K) ((HahnSeries.of …
  -/
  exact ⟨F, by rw [hF, hf, RingEquiv.symm_apply_apply]⟩
  /-
    🎉 no goals
  -/


theorem powerSeries_ext_subring :
    Subring.map (LaurentSeriesRingEquiv K).toRingHom (powerSeries_as_subring K) =
      ((idealX K).adicCompletionIntegers (RatFunc K)).toSubring := by
  /-
    K : Type u_2
    inst✝ : Field K
    ⊢ Eq (Subring.map (LaurentSeries.LaurentSeriesRingEquiv K).toRingHom (LaurentS …
  -/
  ext x
  /-
    case h
    K : Type u_2
    inst✝ : Field K
    x : LaurentSeries.RatFuncAdicCompl K
    ⊢ Iff (Membership.mem (Subring.map (LaurentSeries.LaurentSeriesRingEquiv K).to …
  -/
  refine ⟨fun ⟨f, ⟨F, coe_F⟩, hF⟩ ↦ ?_, fun H ↦ ?_⟩
    /-
      case h.refine_1
      K : Type u_2
      inst✝ : Field K
      x : LaurentSeries.RatFuncAdicCompl K
      x✝ : Membership.mem (Subring.map (LaurentSeries.LaurentSeriesRingEquiv K).toRi …
      f : LaurentSeries K
      F : PowerSeries K
      coe_F : Eq ((HahnSeries.ofPowerSeries Int K) F) f
      hF : Eq ((LaurentSeries.LaurentSeriesRingEquiv K).toRingHom f) x
      ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers (R …
    -/
  · simp only [ValuationSubring.mem_toSubring, ← hF, ← coe_F]
    /-
      case h.refine_1
      K : Type u_2
      inst✝ : Field K
      x : LaurentSeries.RatFuncAdicCompl K
      x✝ : Membership.mem (Subring.map (LaurentSeries.LaurentSeriesRingEquiv K).toRi …
      f : LaurentSeries K
      F : PowerSeries K
      coe_F : Eq ((HahnSeries.ofPowerSeries Int K) F) f
      hF : Eq ((LaurentSeries.LaurentSeriesRingEquiv K).toRingHom f) x
      ⊢ Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers (R …
    -/
    apply mem_integers_of_powerSeries
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      K : Type u_2
      inst✝ : Field K
      x : LaurentSeries.RatFuncAdicCompl K
      H : Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers  …
      ⊢ Membership.mem (Subring.map (LaurentSeries.LaurentSeriesRingEquiv K).toRingH …
    -/
  · obtain ⟨F, hF⟩ := exists_powerSeries_of_memIntegers K H
    simp only [Equiv.toFun_as_coe, UniformEquiv.coe_toEquiv, exists_exists_eq_and,
      UniformEquiv.coe_symm_toEquiv, Subring.mem_map, Equiv.invFun_as_coe]
    /-
      case h.refine_2.intro
      K : Type u_2
      inst✝ : Field K
      x : LaurentSeries.RatFuncAdicCompl K
      H : Membership.mem (IsDedekindDomain.HeightOneSpectrum.adicCompletionIntegers  …
      F : PowerSeries K
      hF : Eq ((LaurentSeries.LaurentSeriesRingEquiv K) ((HahnSeries.ofPowerSeries I …
      ⊢ Exists fun x_1 => And (Membership.mem (LaurentSeries.powerSeries_as_subring  …
    -/
    exact ⟨F, ⟨F, rfl⟩, hF⟩
    /-
      🎉 no goals
    -/


/-- The ring isomorphism between `K⟦X⟧` and the unit ball inside the `X`-adic completion of
`RatFunc K`. -/
abbrev powerSeriesRingEquiv : K⟦X⟧ ≃+* (idealX K).adicCompletionIntegers (RatFunc K) :=
  ((powerSeriesEquivSubring K).trans (LaurentSeriesRingEquiv K).subringMap).trans
    <| RingEquiv.subringCongr (powerSeries_ext_subring K)


