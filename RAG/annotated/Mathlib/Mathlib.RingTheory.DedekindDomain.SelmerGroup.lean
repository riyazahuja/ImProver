local notation K "/" n => Kˣ ⧸ (powMonoidHom n : Kˣ →* Kˣ).range


open Classical in
/-- The multiplicative `v`-adic valuation on `Kˣ`. -/
def valuationOfNeZeroToFun (x : Kˣ) : Multiplicative ℤ :=
  let hx := IsLocalization.sec R⁰ (x : K)
  Multiplicative.ofAdd <|
    (-(Associates.mk v.asIdeal).count (Associates.mk <| Ideal.span {hx.fst}).factors : ℤ) -
      (-(Associates.mk v.asIdeal).count (Associates.mk <| Ideal.span {(hx.snd : R)}).factors : ℤ)


@[simp]
theorem valuationOfNeZeroToFun_eq (x : Kˣ) :
    (v.valuationOfNeZeroToFun x : ℤₘ₀) = v.valuation (x : K) := by
  classical
  rw [show v.valuation (x : K) = _ * _ by rfl]
  rw [Units.val_inv_eq_inv_val]
  change _ = ite _ _ _ * (ite _ _ _)⁻¹
  simp_rw [IsLocalization.toLocalizationMap_sec, SubmonoidClass.coe_subtype,
    if_neg <| IsLocalization.sec_fst_ne_zero le_rfl x.ne_zero,
    if_neg (nonZeroDivisors.coe_ne_zero _),
    valuationOfNeZeroToFun, ofAdd_sub, ofAdd_neg, div_inv_eq_mul, WithZero.coe_mul,
    WithZero.coe_inv, inv_inv]


/-- The multiplicative `v`-adic valuation on `Kˣ`. -/
def valuationOfNeZero : Kˣ →* Multiplicative ℤ where
  toFun := v.valuationOfNeZeroToFun
                 /-
                   R : Type u
                   inst✝⁴ : CommRing R
                   inst✝³ : IsDedekindDomain R
                   K : Type v
                   inst✝² : Field K
                   inst✝¹ : Algebra R K
                   inst✝ : IsFractionRing R K
                   v : IsDedekindDomain.HeightOneSpectrum R
                   ⊢ Eq (v.valuationOfNeZeroToFun 1) 1
                 -/
  map_one' := by rw [← WithZero.coe_inj, valuationOfNeZeroToFun_eq]; exact map_one _
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  map_mul' _ _ := by
    /-
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x✝¹ x✝ : Units K
      ⊢ Eq ({ toFun := v.valuationOfNeZeroToFun, map_one' := ⋯ }.toFun (HMul.hMul x✝ …
    -/
    rw [← WithZero.coe_inj, WithZero.coe_mul]
    /-
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x✝¹ x✝ : Units K
      ⊢ Eq (↑({ toFun := v.valuationOfNeZeroToFun, map_one' := ⋯ }.toFun (HMul.hMul  …
    -/
    simp only [valuationOfNeZeroToFun_eq]; exact map_mul _ _ _
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem valuationOfNeZero_eq (x : Kˣ) : (v.valuationOfNeZero x : ℤₘ₀) = v.valuation (x : K) :=
  valuationOfNeZeroToFun_eq v x


@[simp]
theorem valuation_of_unit_eq (x : Rˣ) :
    v.valuationOfNeZero (Units.map (algebraMap R K : R →* K) x) = 1 := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type v
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    x : Units R
    ⊢ Eq (v.valuationOfNeZero ((Units.map ↑(algebraMap R K)) x)) 1
  -/
  rw [← WithZero.coe_inj, valuationOfNeZero_eq, Units.coe_map, eq_iff_le_not_lt]
  /-
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type v
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    v : IsDedekindDomain.HeightOneSpectrum R
    x : Units R
    ⊢ And (LE.le (v.valuation (↑(algebraMap R K) ↑x)) ↑1) (Not (LT.lt (v.valuation …
  -/
  constructor
    /-
      case left
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x : Units R
      ⊢ LE.le (v.valuation (↑(algebraMap R K) ↑x)) ↑1
    -/
  · exact v.valuation_le_one x
    /-
      🎉 no goals
    -/
    /-
      case right
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x : Units R
      ⊢ Not (LT.lt (v.valuation (↑(algebraMap R K) ↑x)) ↑1)
    -/
  · cases' x with x _ hx _
    /-
      case right.mk
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x inv✝ : R
      hx : Eq (HMul.hMul x inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ x) 1
      ⊢ Not (LT.lt (v.valuation (↑(algebraMap R K) ↑{ val := x, inv := inv✝, val_inv …
    -/
    change ¬v.valuation (algebraMap R K x) < 1
    /-
      case right.mk
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x inv✝ : R
      hx : Eq (HMul.hMul x inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ x) 1
      ⊢ Not (LT.lt (v.valuation ((algebraMap R K) x)) 1)
    -/
    apply_fun v.intValuation at hx
    /-
      case right.mk
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x inv✝ : R
      inv_val✝ : Eq (HMul.hMul inv✝ x) 1
      hx : Eq (v.intValuation (HMul.hMul x inv✝)) (v.intValuation 1)
      ⊢ Not (LT.lt (v.valuation ((algebraMap R K) x)) 1)
    -/
    rw [map_one, map_mul] at hx
    rw [not_lt, ← hx, ← mul_one <| v.valuation _, valuation_of_algebraMap,
      mul_le_mul_left <| zero_lt_iff.2 <| left_ne_zero_of_mul_eq_one hx]
    /-
      case right.mk
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      v : IsDedekindDomain.HeightOneSpectrum R
      x inv✝ : R
      inv_val✝ : Eq (HMul.hMul inv✝ x) 1
      hx : Eq (HMul.hMul (v.intValuation x) (v.intValuation inv✝)) 1
      ⊢ LE.le (v.intValuation inv✝) 1
    -/
    exact v.intValuation_le_one _
    /-
      🎉 no goals
    -/

-- Porting note: invalid attribute 'semireducible', declaration is in an imported module
-- attribute [local semireducible] MulOpposite


/-- The multiplicative `v`-adic valuation on `Kˣ` modulo `n`-th powers. -/
def valuationOfNeZeroMod (n : ℕ) : (K/n) →* Multiplicative (ZMod n) :=
  (Int.quotientZMultiplesNatEquivZMod n).toMultiplicative.toMonoidHom.comp <|
    QuotientGroup.map (powMonoidHom n : Kˣ →* Kˣ).range
      (AddSubgroup.toSubgroup (AddSubgroup.zmultiples (n : ℤ)))
      v.valuationOfNeZero
      (by
        /-
          R : Type u
          inst✝⁴ : CommRing R
          inst✝³ : IsDedekindDomain R
          K : Type v
          inst✝² : Field K
          inst✝¹ : Algebra R K
          inst✝ : IsFractionRing R K
          v : IsDedekindDomain.HeightOneSpectrum R
          n : Nat
          ⊢ LE.le (powMonoidHom n).range (Subgroup.comap v.valuationOfNeZero (AddSubgrou …
        -/
        rintro _ ⟨x, rfl⟩
        exact
          ⟨v.valuationOfNeZero x, by simp only [powMonoidHom_apply, map_pow, Int.toAdd_pow]; rfl⟩)


@[simp]
theorem valuation_of_unit_mod_eq (n : ℕ) (x : Rˣ) :
    v.valuationOfNeZeroMod n (Units.map (algebraMap R K : R →* K) x : K/n) = 1 := by
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  erw [valuationOfNeZeroMod, MonoidHom.comp_apply, ← QuotientGroup.coe_mk',
    QuotientGroup.map_mk' (G := Kˣ) (N := MonoidHom.range (powMonoidHom n)),
    valuation_of_unit_eq, QuotientGroup.mk_one, map_one]


/-- The Selmer group `K⟮S, n⟯`. -/
def selmerGroup : Subgroup <| K/n where
  carrier := {x : K/n | ∀ (v) (_ : v ∉ S), (v : HeightOneSpectrum R).valuationOfNeZeroMod n x = 1}
                     /-
                       R : Type u
                       inst✝⁴ : CommRing R
                       inst✝³ : IsDedekindDomain R
                       K : Type v
                       inst✝² : Field K
                       inst✝¹ : Algebra R K
                       inst✝ : IsFractionRing R K
                       v : IsDedekindDomain.HeightOneSpectrum R
                       S S' : Set (IsDedekindDomain.HeightOneSpectrum R)
                       n : Nat
                       x✝¹ : IsDedekindDomain.HeightOneSpectrum R
                       x✝ : Not (Membership.mem S x✝¹)
                       ⊢ Eq ((x✝¹.valuationOfNeZeroMod n) 1) 1
                     -/
                            /-
                              R : Type u
                              inst✝⁴ : CommRing R
                              inst✝³ : IsDedekindDomain R
                              K : Type v
                              inst✝² : Field K
                              inst✝¹ : Algebra R K
                              inst✝ : IsFractionRing R K
                              v✝ : IsDedekindDomain.HeightOneSpectrum R
                              S S' : Set (IsDedekindDomain.HeightOneSpectrum R)
                              n : Nat
                              a✝ b✝ : HasQuotient.Quotient (Units K) (powMonoidHom n).range
                              hx : Membership.mem (setOf fun x => ∀ (v : IsDedekindDomain.HeightOneSpectrum  …
                              hy : Membership.mem (setOf fun x => ∀ (v : IsDedekindDomain.HeightOneSpectrum  …
                              v : IsDedekindDomain.HeightOneSpectrum R
                              hv : Not (Membership.mem S v)
                              ⊢ Eq ((v.valuationOfNeZeroMod n) (HMul.hMul a✝ b✝)) 1
                            -/
  one_mem' _ _ := by rw [map_one]
                            /-
                              🎉 no goals
                            -/
                     /-
                       🎉 no goals
                     -/
  mul_mem' hx hy v hv := by rw [map_mul, hx v hv, hy v hv, one_mul]
                         /-
                           R : Type u
                           inst✝⁴ : CommRing R
                           inst✝³ : IsDedekindDomain R
                           K : Type v
                           inst✝² : Field K
                           inst✝¹ : Algebra R K
                           inst✝ : IsFractionRing R K
                           v✝ : IsDedekindDomain.HeightOneSpectrum R
                           S S' : Set (IsDedekindDomain.HeightOneSpectrum R)
                           n : Nat
                           x✝ : HasQuotient.Quotient (Units K) (powMonoidHom n).range
                           hx : Membership.mem { carrier := setOf fun x => ∀ (v : IsDedekindDomain.Height …
                           v : IsDedekindDomain.HeightOneSpectrum R
                           hv : Not (Membership.mem S v)
                           ⊢ Eq ((v.valuationOfNeZeroMod n) (Inv.inv x✝)) 1
                         -/
  inv_mem' hx v hv := by rw [map_inv, hx v hv, inv_one]
                         /-
                           🎉 no goals
                         -/

-- Porting note: was `scoped[SelmerGroup]` but that does not work even using `open SelmerGroup`

local notation K "⟮" S "," n "⟯" => @selmerGroup _ _ _ K _ _ _ S n


theorem monotone (hS : S ≤ S') : K⟮S,n⟯ ≤ K⟮S',n⟯ := fun _ hx v => hx v ∘ mt (@hS v)


/-- The multiplicative `v`-adic valuations on `K⟮S, n⟯` for all `v ∈ S`. -/
def valuation : K⟮S,n⟯ →* S → Multiplicative (ZMod n) where
  toFun x v := (v : HeightOneSpectrum R).valuationOfNeZeroMod n (x : K/n)
  map_one' := funext fun _ => map_one _
                     /-
                       R : Type u
                       inst✝⁴ : CommRing R
                       inst✝³ : IsDedekindDomain R
                       K : Type v
                       inst✝² : Field K
                       inst✝¹ : Algebra R K
                       inst✝ : IsFractionRing R K
                       v : IsDedekindDomain.HeightOneSpectrum R
                       S S' : Set (IsDedekindDomain.HeightOneSpectrum R)
                       n : Nat
                       x y : Subtype fun x => Membership.mem IsDedekindDomain.selmerGroup x
                       ⊢ Eq ({ toFun := fun x v => ((↑v).valuationOfNeZeroMod n) ↑x, map_one' := ⋯ }. …
                     -/
  map_mul' x y := by simp only [Subgroup.coe_mul, map_mul]; rfl
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem valuation_ker_eq :
    valuation.ker = K⟮(∅ : Set <| HeightOneSpectrum R),n⟯.subgroupOf (K⟮S,n⟯) := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type v
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    S : Set (IsDedekindDomain.HeightOneSpectrum R)
    n : Nat
    ⊢ Eq IsDedekindDomain.selmerGroup.valuation.ker (IsDedekindDomain.selmerGroup. …
  -/
  ext ⟨_, hx⟩
  /-
    case h.mk
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type v
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    S : Set (IsDedekindDomain.HeightOneSpectrum R)
    n : Nat
    val✝ : HasQuotient.Quotient (Units K) (powMonoidHom n).range
    hx : Membership.mem IsDedekindDomain.selmerGroup val✝
    ⊢ Iff (Membership.mem IsDedekindDomain.selmerGroup.valuation.ker ⟨val✝, hx⟩) ( …
  -/
  constructor
    /-
      case h.mk.mp
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      S : Set (IsDedekindDomain.HeightOneSpectrum R)
      n : Nat
      val✝ : HasQuotient.Quotient (Units K) (powMonoidHom n).range
      hx : Membership.mem IsDedekindDomain.selmerGroup val✝
      ⊢ Membership.mem IsDedekindDomain.selmerGroup.valuation.ker ⟨val✝, hx⟩ → Membe …
    -/
  · intro hx' v _
    /-
      case h.mk.mp
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      S : Set (IsDedekindDomain.HeightOneSpectrum R)
      n : Nat
      val✝ : HasQuotient.Quotient (Units K) (powMonoidHom n).range
      hx : Membership.mem IsDedekindDomain.selmerGroup val✝
      hx' : Membership.mem IsDedekindDomain.selmerGroup.valuation.ker ⟨val✝, hx⟩
      v : IsDedekindDomain.HeightOneSpectrum R
      x✝ : Not (Membership.mem EmptyCollection.emptyCollection v)
      ⊢ Eq ((v.valuationOfNeZeroMod n) (IsDedekindDomain.selmerGroup.subtype ⟨val✝,  …
    -/
    by_cases hv : v ∈ S
      /-
        case pos
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        K : Type v
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        S : Set (IsDedekindDomain.HeightOneSpectrum R)
        n : Nat
        val✝ : HasQuotient.Quotient (Units K) (powMonoidHom n).range
        hx : Membership.mem IsDedekindDomain.selmerGroup val✝
        hx' : Membership.mem IsDedekindDomain.selmerGroup.valuation.ker ⟨val✝, hx⟩
        v : IsDedekindDomain.HeightOneSpectrum R
        x✝ : Not (Membership.mem EmptyCollection.emptyCollection v)
        hv : Membership.mem S v
        ⊢ Eq ((v.valuationOfNeZeroMod n) (IsDedekindDomain.selmerGroup.subtype ⟨val✝,  …
      -/
    · exact congr_fun hx' ⟨v, hv⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        K : Type v
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        S : Set (IsDedekindDomain.HeightOneSpectrum R)
        n : Nat
        val✝ : HasQuotient.Quotient (Units K) (powMonoidHom n).range
        hx : Membership.mem IsDedekindDomain.selmerGroup val✝
        hx' : Membership.mem IsDedekindDomain.selmerGroup.valuation.ker ⟨val✝, hx⟩
        v : IsDedekindDomain.HeightOneSpectrum R
        x✝ : Not (Membership.mem EmptyCollection.emptyCollection v)
        hv : Not (Membership.mem S v)
        ⊢ Eq ((v.valuationOfNeZeroMod n) (IsDedekindDomain.selmerGroup.subtype ⟨val✝,  …
      -/
    · exact hx v hv
      /-
        🎉 no goals
      -/
    /-
      case h.mk.mpr
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      S : Set (IsDedekindDomain.HeightOneSpectrum R)
      n : Nat
      val✝ : HasQuotient.Quotient (Units K) (powMonoidHom n).range
      hx : Membership.mem IsDedekindDomain.selmerGroup val✝
      ⊢ Membership.mem (IsDedekindDomain.selmerGroup.subgroupOf IsDedekindDomain.sel …
    -/
  · exact fun hx' => funext fun v => hx' v <| Set.not_mem_empty v
    /-
      🎉 no goals
    -/


/-- The natural homomorphism from `Rˣ` to `K⟮∅, n⟯`. -/
def fromUnit {n : ℕ} : Rˣ →* K⟮(∅ : Set <| HeightOneSpectrum R),n⟯ where
  toFun x :=
    ⟨QuotientGroup.mk <| Units.map (algebraMap R K).toMonoidHom x, fun v _ =>
      v.valuation_of_unit_mod_eq n x⟩
                 /-
                   R : Type u
                   inst✝⁴ : CommRing R
                   inst✝³ : IsDedekindDomain R
                   K : Type v
                   inst✝² : Field K
                   inst✝¹ : Algebra R K
                   inst✝ : IsFractionRing R K
                   v : IsDedekindDomain.HeightOneSpectrum R
                   S S' : Set (IsDedekindDomain.HeightOneSpectrum R)
                   n✝ n : Nat
                   ⊢ Eq ((fun x => ⟨↑((Units.map ↑(algebraMap R K)) x), ⋯⟩) 1) 1
                 -/
  map_one' := by simp only [map_one, QuotientGroup.mk_one, Subgroup.mk_eq_one]
                 /-
                   🎉 no goals
                 -/
  map_mul' _ _ := by simp only [RingHom.toMonoidHom_eq_coe, map_mul, QuotientGroup.mk_mul,
    MulMemClass.mk_mul_mk]


theorem fromUnit_ker [hn : Fact <| 0 < n] :
    (@fromUnit R _ _ K _ _ _ n).ker = (powMonoidHom n : Rˣ →* Rˣ).range := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type v
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    n : Nat
    hn : Fact (LT.lt 0 n)
    ⊢ Eq IsDedekindDomain.selmerGroup.fromUnit.ker (powMonoidHom n).range
  -/
  ext ⟨_, _, _, _⟩
  /-
    case h.mk
    R : Type u
    inst✝⁴ : CommRing R
    inst✝³ : IsDedekindDomain R
    K : Type v
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    n : Nat
    hn : Fact (LT.lt 0 n)
    val✝ inv✝ : R
    val_inv✝ : Eq (HMul.hMul val✝ inv✝) 1
    inv_val✝ : Eq (HMul.hMul inv✝ val✝) 1
    ⊢ Iff (Membership.mem IsDedekindDomain.selmerGroup.fromUnit.ker { val := val✝, …
  -/
  constructor
    /-
      case h.mk.mp
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      n : Nat
      hn : Fact (LT.lt 0 n)
      val✝ inv✝ : R
      val_inv✝ : Eq (HMul.hMul val✝ inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ val✝) 1
      ⊢ Membership.mem IsDedekindDomain.selmerGroup.fromUnit.ker { val := val✝, inv  …
    -/
  · intro hx
    /-
      case h.mk.mp
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      n : Nat
      hn : Fact (LT.lt 0 n)
      val✝ inv✝ : R
      val_inv✝ : Eq (HMul.hMul val✝ inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ val✝) 1
      hx : Membership.mem IsDedekindDomain.selmerGroup.fromUnit.ker { val := val✝, i …
      ⊢ Membership.mem (powMonoidHom n).range { val := val✝, inv := inv✝, val_inv := …
    -/
    rcases (QuotientGroup.eq_one_iff _).mp (Subtype.mk.inj hx) with ⟨⟨v, i, vi, iv⟩, hx⟩
    /-
      case h.mk.mp.intro.mk
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      n : Nat
      hn : Fact (LT.lt 0 n)
      val✝ inv✝ : R
      val_inv✝ : Eq (HMul.hMul val✝ inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ val✝) 1
      hx✝ : Membership.mem IsDedekindDomain.selmerGroup.fromUnit.ker { val := val✝,  …
      v i : K
      vi : Eq (HMul.hMul v i) 1
      iv : Eq (HMul.hMul i v) 1
      hx : Eq ((powMonoidHom n) { val := v, inv := i, val_inv := vi, inv_val := iv } …
      ⊢ Membership.mem (powMonoidHom n).range { val := val✝, inv := inv✝, val_inv := …
    -/
    have hv : ↑(_ ^ n : Kˣ) = algebraMap R K _ := congr_arg Units.val hx
    /-
      case h.mk.mp.intro.mk
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      n : Nat
      hn : Fact (LT.lt 0 n)
      val✝ inv✝ : R
      val_inv✝ : Eq (HMul.hMul val✝ inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ val✝) 1
      hx✝ : Membership.mem IsDedekindDomain.selmerGroup.fromUnit.ker { val := val✝,  …
      v i : K
      vi : Eq (HMul.hMul v i) 1
      iv : Eq (HMul.hMul i v) 1
      hx : Eq ((powMonoidHom n) { val := v, inv := i, val_inv := vi, inv_val := iv } …
      hv : Eq (↑(HPow.hPow { val := v, inv := i, val_inv := vi, inv_val := iv } n))  …
      ⊢ Membership.mem (powMonoidHom n).range { val := val✝, inv := inv✝, val_inv := …
    -/
    have hi : ↑(_ ^ n : Kˣ)⁻¹ = algebraMap R K _ := congr_arg Units.inv hx
    /-
      case h.mk.mp.intro.mk
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      n : Nat
      hn : Fact (LT.lt 0 n)
      val✝ inv✝ : R
      val_inv✝ : Eq (HMul.hMul val✝ inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ val✝) 1
      hx✝ : Membership.mem IsDedekindDomain.selmerGroup.fromUnit.ker { val := val✝,  …
      v i : K
      vi : Eq (HMul.hMul v i) 1
      iv : Eq (HMul.hMul i v) 1
      hx : Eq ((powMonoidHom n) { val := v, inv := i, val_inv := vi, inv_val := iv } …
      hv : Eq (↑(HPow.hPow { val := v, inv := i, val_inv := vi, inv_val := iv } n))  …
      hi : Eq (↑(Inv.inv (HPow.hPow { val := v, inv := i, val_inv := vi, inv_val :=  …
      ⊢ Membership.mem (powMonoidHom n).range { val := val✝, inv := inv✝, val_inv := …
    -/
    rw [Units.val_pow_eq_pow_val] at hv
    /-
      case h.mk.mp.intro.mk
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      n : Nat
      hn : Fact (LT.lt 0 n)
      val✝ inv✝ : R
      val_inv✝ : Eq (HMul.hMul val✝ inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ val✝) 1
      hx✝ : Membership.mem IsDedekindDomain.selmerGroup.fromUnit.ker { val := val✝,  …
      v i : K
      vi : Eq (HMul.hMul v i) 1
      iv : Eq (HMul.hMul i v) 1
      hx : Eq ((powMonoidHom n) { val := v, inv := i, val_inv := vi, inv_val := iv } …
      hv : Eq (HPow.hPow (↑{ val := v, inv := i, val_inv := vi, inv_val := iv }) n)  …
      hi : Eq (↑(Inv.inv (HPow.hPow { val := v, inv := i, val_inv := vi, inv_val :=  …
      ⊢ Membership.mem (powMonoidHom n).range { val := val✝, inv := inv✝, val_inv := …
    -/
    rw [← inv_pow, Units.inv_mk, Units.val_pow_eq_pow_val] at hi
    rcases IsIntegrallyClosed.exists_algebraMap_eq_of_isIntegral_pow (R := R) (x := v) hn.out
        (hv.symm ▸ isIntegral_algebraMap) with
      ⟨v', rfl⟩
    rcases IsIntegrallyClosed.exists_algebraMap_eq_of_isIntegral_pow (R := R) (x := i) hn.out
        (hi.symm ▸ isIntegral_algebraMap) with
      ⟨i', rfl⟩
    /-
      case h.mk.mp.intro.mk.intro.intro
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      n : Nat
      hn : Fact (LT.lt 0 n)
      val✝ inv✝ : R
      val_inv✝ : Eq (HMul.hMul val✝ inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ val✝) 1
      hx✝ : Membership.mem IsDedekindDomain.selmerGroup.fromUnit.ker { val := val✝,  …
      v' i' : R
      vi : Eq (HMul.hMul ((algebraMap R K) v') ((algebraMap R K) i')) 1
      iv : Eq (HMul.hMul ((algebraMap R K) i') ((algebraMap R K) v')) 1
      hx : Eq ((powMonoidHom n) { val := (algebraMap R K) v', inv := (algebraMap R K …
      hv : Eq (HPow.hPow (↑{ val := (algebraMap R K) v', inv := (algebraMap R K) i', …
      hi : Eq (HPow.hPow (↑{ val := (algebraMap R K) i', inv := (algebraMap R K) v', …
      ⊢ Membership.mem (powMonoidHom n).range { val := val✝, inv := inv✝, val_inv := …
    -/
    rw [← map_mul, map_eq_one_iff _ <| NoZeroSMulDivisors.algebraMap_injective R K] at vi
    /-
      case h.mk.mp.intro.mk.intro.intro
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      n : Nat
      hn : Fact (LT.lt 0 n)
      val✝ inv✝ : R
      val_inv✝ : Eq (HMul.hMul val✝ inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ val✝) 1
      hx✝ : Membership.mem IsDedekindDomain.selmerGroup.fromUnit.ker { val := val✝,  …
      v' i' : R
      vi✝ : Eq (HMul.hMul ((algebraMap R K) v') ((algebraMap R K) i')) 1
      vi : Eq (HMul.hMul v' i') 1
      iv : Eq (HMul.hMul ((algebraMap R K) i') ((algebraMap R K) v')) 1
      hx : Eq ((powMonoidHom n) { val := (algebraMap R K) v', inv := (algebraMap R K …
      hv : Eq (HPow.hPow (↑{ val := (algebraMap R K) v', inv := (algebraMap R K) i', …
      hi : Eq (HPow.hPow (↑{ val := (algebraMap R K) i', inv := (algebraMap R K) v', …
      ⊢ Membership.mem (powMonoidHom n).range { val := val✝, inv := inv✝, val_inv := …
    -/
    rw [← map_mul, map_eq_one_iff _ <| NoZeroSMulDivisors.algebraMap_injective R K] at iv
    /-
      case h.mk.mp.intro.mk.intro.intro
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      n : Nat
      hn : Fact (LT.lt 0 n)
      val✝ inv✝ : R
      val_inv✝ : Eq (HMul.hMul val✝ inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ val✝) 1
      hx✝ : Membership.mem IsDedekindDomain.selmerGroup.fromUnit.ker { val := val✝,  …
      v' i' : R
      vi✝ : Eq (HMul.hMul ((algebraMap R K) v') ((algebraMap R K) i')) 1
      vi : Eq (HMul.hMul v' i') 1
      iv✝ : Eq (HMul.hMul ((algebraMap R K) i') ((algebraMap R K) v')) 1
      iv : Eq (HMul.hMul i' v') 1
      hx : Eq ((powMonoidHom n) { val := (algebraMap R K) v', inv := (algebraMap R K …
      hv : Eq (HPow.hPow (↑{ val := (algebraMap R K) v', inv := (algebraMap R K) i', …
      hi : Eq (HPow.hPow (↑{ val := (algebraMap R K) i', inv := (algebraMap R K) v', …
      ⊢ Membership.mem (powMonoidHom n).range { val := val✝, inv := inv✝, val_inv := …
    -/
    rw [Units.val_mk, ← map_pow] at hv
    exact ⟨⟨v', i', vi, iv⟩, by
      simpa only [Units.ext_iff, powMonoidHom_apply, Units.val_pow_eq_pow_val] using
         NoZeroSMulDivisors.algebraMap_injective R K hv⟩
    /-
      case h.mk.mpr
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      n : Nat
      hn : Fact (LT.lt 0 n)
      val✝ inv✝ : R
      val_inv✝ : Eq (HMul.hMul val✝ inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ val✝) 1
      ⊢ Membership.mem (powMonoidHom n).range { val := val✝, inv := inv✝, val_inv := …
    -/
  · rintro ⟨x, hx⟩
    /-
      case h.mk.mpr.intro
      R : Type u
      inst✝⁴ : CommRing R
      inst✝³ : IsDedekindDomain R
      K : Type v
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      n : Nat
      hn : Fact (LT.lt 0 n)
      val✝ inv✝ : R
      val_inv✝ : Eq (HMul.hMul val✝ inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ val✝) 1
      x : Units R
      hx : Eq ((powMonoidHom n) x) { val := val✝, inv := inv✝, val_inv := val_inv✝,  …
      ⊢ Membership.mem IsDedekindDomain.selmerGroup.fromUnit.ker { val := val✝, inv  …
    -/
    rw [← hx]
    exact Subtype.mk_eq_mk.mpr <| (QuotientGroup.eq_one_iff _).mpr ⟨Units.map (algebraMap R K) x,
      by simp only [powMonoidHom_apply, RingHom.toMonoidHom_eq_coe, map_pow]⟩


/-- The injection induced by the natural homomorphism from `Rˣ` to `K⟮∅, n⟯`. -/
def fromUnitLift [Fact <| 0 < n] : (R/n) →* K⟮(∅ : Set <| HeightOneSpectrum R),n⟯ :=
  (QuotientGroup.kerLift _).comp
    (QuotientGroup.quotientMulEquivOfEq (fromUnit_ker (R := R))).symm.toMonoidHom


theorem fromUnitLift_injective [Fact <| 0 < n] :
    Function.Injective <| @fromUnitLift R _ _ K _ _ _ n _ := by
  /-
    R : Type u
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDedekindDomain R
    K : Type v
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    n : Nat
    inst✝ : Fact (LT.lt 0 n)
    ⊢ Function.Injective ⇑IsDedekindDomain.selmerGroup.fromUnitLift
  -/
  dsimp only [fromUnitLift, MonoidHom.coe_comp, MulEquiv.coe_toMonoidHom]
  /-
    R : Type u
    inst✝⁵ : CommRing R
    inst✝⁴ : IsDedekindDomain R
    K : Type v
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    n : Nat
    inst✝ : Fact (LT.lt 0 n)
    ⊢ Function.Injective (Function.comp ⇑(QuotientGroup.kerLift IsDedekindDomain.s …
  -/
  exact Function.Injective.comp (QuotientGroup.kerLift_injective _) (MulEquiv.injective _)
  /-
    🎉 no goals
  -/


