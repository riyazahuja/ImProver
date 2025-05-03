/-- `R` is integrally closed in `A` if all integral elements of `A` are also elements of `R`.
-/
abbrev IsIntegrallyClosedIn (R A : Type*) [CommRing R] [CommRing A] [Algebra R A] :=
  IsIntegralClosure R R A


/-- `R` is integrally closed if all integral elements of `Frac(R)` are also elements of `R`.

This definition uses `FractionRing R` to denote `Frac(R)`. See `isIntegrallyClosed_iff`
if you want to choose another field of fractions for `R`.
-/
abbrev IsIntegrallyClosed (R : Type*) [CommRing R] := IsIntegrallyClosedIn R (FractionRing R)


/-- Being integrally closed is preserved under injective algebra homomorphisms. -/
theorem AlgHom.isIntegrallyClosedIn (f : A →ₐ[R] B) (hf : Function.Injective f) :
    IsIntegrallyClosedIn R B → IsIntegrallyClosedIn R A := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f : AlgHom R A B
    hf : Function.Injective ⇑f
    ⊢ IsIntegrallyClosedIn R B → IsIntegrallyClosedIn R A
  -/
  rintro ⟨inj, cl⟩
  /-
    case mk
    R : Type u_1
    inst✝⁴ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f : AlgHom R A B
    hf : Function.Injective ⇑f
    inj : Function.Injective ⇑(algebraMap R B)
    cl : ∀ {x : B}, Iff (IsIntegral R x) (Exists fun y => Eq ((algebraMap R B) y) x)
    ⊢ IsIntegrallyClosedIn R A
  -/
  refine ⟨Function.Injective.of_comp (f := f) ?_, fun hx => ?_, ?_⟩
    /-
      case mk.refine_1
      R : Type u_1
      inst✝⁴ : CommRing R
      A : Type u_2
      B : Type u_3
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f : AlgHom R A B
      hf : Function.Injective ⇑f
      inj : Function.Injective ⇑(algebraMap R B)
      cl : ∀ {x : B}, Iff (IsIntegral R x) (Exists fun y => Eq ((algebraMap R B) y) x)
      ⊢ Function.Injective (Function.comp ⇑f ⇑(algebraMap R A))
    -/
  · convert inj
    /-
      case h.e'_3
      R : Type u_1
      inst✝⁴ : CommRing R
      A : Type u_2
      B : Type u_3
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f : AlgHom R A B
      hf : Function.Injective ⇑f
      inj : Function.Injective ⇑(algebraMap R B)
      cl : ∀ {x : B}, Iff (IsIntegral R x) (Exists fun y => Eq ((algebraMap R B) y) x)
      ⊢ Eq (Function.comp ⇑f ⇑(algebraMap R A)) ⇑(algebraMap R B)
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      case mk.refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      A : Type u_2
      B : Type u_3
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f : AlgHom R A B
      hf : Function.Injective ⇑f
      inj : Function.Injective ⇑(algebraMap R B)
      cl : ∀ {x : B}, Iff (IsIntegral R x) (Exists fun y => Eq ((algebraMap R B) y) x)
      x✝ : A
      hx : IsIntegral R x✝
      ⊢ Exists fun y => Eq ((algebraMap R A) y) x✝
    -/
  · obtain ⟨y, fx_eq⟩ := cl.mp ((isIntegral_algHom_iff f hf).mpr hx)
    /-
      case mk.refine_2.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      A : Type u_2
      B : Type u_3
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f : AlgHom R A B
      hf : Function.Injective ⇑f
      inj : Function.Injective ⇑(algebraMap R B)
      cl : ∀ {x : B}, Iff (IsIntegral R x) (Exists fun y => Eq ((algebraMap R B) y) x)
      x✝ : A
      hx : IsIntegral R x✝
      y : R
      fx_eq : Eq ((algebraMap R B) y) (f x✝)
      ⊢ Exists fun y => Eq ((algebraMap R A) y) x✝
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      case mk.refine_3
      R : Type u_1
      inst✝⁴ : CommRing R
      A : Type u_2
      B : Type u_3
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f : AlgHom R A B
      hf : Function.Injective ⇑f
      inj : Function.Injective ⇑(algebraMap R B)
      cl : ∀ {x : B}, Iff (IsIntegral R x) (Exists fun y => Eq ((algebraMap R B) y) x)
      x✝ : A
      ⊢ (Exists fun y => Eq ((algebraMap R A) y) x✝) → IsIntegral R x✝
    -/
  · rintro ⟨y, rfl⟩
    /-
      case mk.refine_3.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      A : Type u_2
      B : Type u_3
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f : AlgHom R A B
      hf : Function.Injective ⇑f
      inj : Function.Injective ⇑(algebraMap R B)
      cl : ∀ {x : B}, Iff (IsIntegral R x) (Exists fun y => Eq ((algebraMap R B) y) x)
      y : R
      ⊢ IsIntegral R ((algebraMap R A) y)
    -/
    apply (isIntegral_algHom_iff f hf).mp
    /-
      case mk.refine_3.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      A : Type u_2
      B : Type u_3
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      f : AlgHom R A B
      hf : Function.Injective ⇑f
      inj : Function.Injective ⇑(algebraMap R B)
      cl : ∀ {x : B}, Iff (IsIntegral R x) (Exists fun y => Eq ((algebraMap R B) y) x)
      y : R
      ⊢ IsIntegral R (f ((algebraMap R A) y))
    -/
    aesop
    /-
      🎉 no goals
    -/


/-- Being integrally closed is preserved under algebra isomorphisms. -/
theorem AlgEquiv.isIntegrallyClosedIn (e : A ≃ₐ[R] B) :
    IsIntegrallyClosedIn R A ↔ IsIntegrallyClosedIn R B :=
  ⟨AlgHom.isIntegrallyClosedIn e.symm e.symm.injective, AlgHom.isIntegrallyClosedIn e e.injective⟩


/-- `R` is integrally closed iff it is the integral closure of itself in its field of fractions. -/
theorem isIntegrallyClosed_iff_isIntegrallyClosedIn :
    IsIntegrallyClosed R ↔ IsIntegrallyClosedIn R K :=
  (IsLocalization.algEquiv R⁰ _ _).isIntegrallyClosedIn


/-- `R` is integrally closed iff it is the integral closure of itself in its field of fractions. -/
theorem isIntegrallyClosed_iff_isIntegralClosure : IsIntegrallyClosed R ↔ IsIntegralClosure R R K :=
  isIntegrallyClosed_iff_isIntegrallyClosedIn K


/-- `R` is integrally closed in `A` iff all integral elements of `A` are also elements of `R`. -/
theorem isIntegrallyClosedIn_iff {R A : Type*} [CommRing R] [CommRing A] [Algebra R A] :
    IsIntegrallyClosedIn R A ↔
      Function.Injective (algebraMap R A) ∧
        ∀ {x : A}, IsIntegral R x → ∃ y, algebraMap R A y = x := by
  /-
    R : Type u_5
    A : Type u_6
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Iff (IsIntegrallyClosedIn R A) (And (Function.Injective ⇑(algebraMap R A)) ( …
  -/
  constructor
    /-
      case mp
      R : Type u_5
      A : Type u_6
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ⊢ IsIntegrallyClosedIn R A → And (Function.Injective ⇑(algebraMap R A)) (∀ {x  …
    -/
  · rintro ⟨_, cl⟩
    /-
      case mp.mk
      R : Type u_5
      A : Type u_6
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      algebraMap_injective'✝ : Function.Injective ⇑(algebraMap R A)
      cl : ∀ {x : A}, Iff (IsIntegral R x) (Exists fun y => Eq ((algebraMap R A) y) x)
      ⊢ And (Function.Injective ⇑(algebraMap R A)) (∀ {x : A}, IsIntegral R x → Exis …
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_5
      A : Type u_6
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ⊢ And (Function.Injective ⇑(algebraMap R A)) (∀ {x : A}, IsIntegral R x → Exis …
    -/
  · rintro ⟨inj, cl⟩
    /-
      case mpr.intro
      R : Type u_5
      A : Type u_6
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      inj : Function.Injective ⇑(algebraMap R A)
      cl : ∀ {x : A}, IsIntegral R x → Exists fun y => Eq ((algebraMap R A) y) x
      ⊢ IsIntegrallyClosedIn R A
    -/
    refine ⟨inj, by aesop, ?_⟩
    /-
      case mpr.intro
      R : Type u_5
      A : Type u_6
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      inj : Function.Injective ⇑(algebraMap R A)
      cl : ∀ {x : A}, IsIntegral R x → Exists fun y => Eq ((algebraMap R A) y) x
      x✝ : A
      ⊢ (Exists fun y => Eq ((algebraMap R A) y) x✝) → IsIntegral R x✝
    -/
    rintro ⟨y, rfl⟩
    /-
      case mpr.intro.intro
      R : Type u_5
      A : Type u_6
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      inj : Function.Injective ⇑(algebraMap R A)
      cl : ∀ {x : A}, IsIntegral R x → Exists fun y => Eq ((algebraMap R A) y) x
      y : R
      ⊢ IsIntegral R ((algebraMap R A) y)
    -/
    apply isIntegral_algebraMap
    /-
      🎉 no goals
    -/


/-- `R` is integrally closed iff all integral elements of its fraction field `K`
are also elements of `R`. -/
theorem isIntegrallyClosed_iff :
    IsIntegrallyClosed R ↔ ∀ {x : K}, IsIntegral R x → ∃ y, algebraMap R K y = x := by
  simp [isIntegrallyClosed_iff_isIntegrallyClosedIn K, isIntegrallyClosedIn_iff,
        IsFractionRing.injective R K]


theorem algebraMap_eq_of_integral [IsIntegrallyClosedIn R A] {x : A} :
    IsIntegral R x → ∃ y : R, algebraMap R A y = x :=
  IsIntegralClosure.isIntegral_iff.mp


theorem isIntegral_iff [IsIntegrallyClosedIn R A] {x : A} :
    IsIntegral R x ↔ ∃ y : R, algebraMap R A y = x :=
  IsIntegralClosure.isIntegral_iff


theorem exists_algebraMap_eq_of_isIntegral_pow [IsIntegrallyClosedIn R A]
    {x : A} {n : ℕ} (hn : 0 < n)
    (hx : IsIntegral R <| x ^ n) : ∃ y : R, algebraMap R A y = x :=
  isIntegral_iff.mp <| hx.of_pow hn


theorem exists_algebraMap_eq_of_pow_mem_subalgebra {A : Type*} [CommRing A] [Algebra R A]
    {S : Subalgebra R A} [IsIntegrallyClosedIn S A] {x : A} {n : ℕ} (hn : 0 < n)
    (hx : x ^ n ∈ S) : ∃ y : S, algebraMap S A y = x :=
  exists_algebraMap_eq_of_isIntegral_pow hn <| isIntegral_iff.mpr ⟨⟨x ^ n, hx⟩, rfl⟩


theorem integralClosure_eq_bot_iff (hRA : Function.Injective (algebraMap R A)) :
    integralClosure R A = ⊥ ↔ IsIntegrallyClosedIn R A := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hRA : Function.Injective ⇑(algebraMap R A)
    ⊢ Iff (Eq (integralClosure R A) Bot.bot) (IsIntegrallyClosedIn R A)
  -/
  refine eq_bot_iff.trans ?_
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hRA : Function.Injective ⇑(algebraMap R A)
    ⊢ Iff (LE.le (integralClosure R A) Bot.bot) (IsIntegrallyClosedIn R A)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      A : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      hRA : Function.Injective ⇑(algebraMap R A)
      ⊢ LE.le (integralClosure R A) Bot.bot → IsIntegrallyClosedIn R A
    -/
  · intro h
    /-
      case mp
      R : Type u_1
      A : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      hRA : Function.Injective ⇑(algebraMap R A)
      h : LE.le (integralClosure R A) Bot.bot
      ⊢ IsIntegrallyClosedIn R A
    -/
    refine ⟨ hRA, fun hx => Set.mem_range.mp (Algebra.mem_bot.mp (h hx)), ?_⟩
    /-
      case mp
      R : Type u_1
      A : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      hRA : Function.Injective ⇑(algebraMap R A)
      h : LE.le (integralClosure R A) Bot.bot
      x✝ : A
      ⊢ (Exists fun y => Eq ((algebraMap R A) y) x✝) → IsIntegral R x✝
    -/
    rintro ⟨y, rfl⟩
    /-
      case mp.intro
      R : Type u_1
      A : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      hRA : Function.Injective ⇑(algebraMap R A)
      h : LE.le (integralClosure R A) Bot.bot
      y : R
      ⊢ IsIntegral R ((algebraMap R A) y)
    -/
    apply isIntegral_algebraMap
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      A : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      hRA : Function.Injective ⇑(algebraMap R A)
      ⊢ IsIntegrallyClosedIn R A → LE.le (integralClosure R A) Bot.bot
    -/
  · intro h x hx
    /-
      case mpr
      R : Type u_1
      A : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      hRA : Function.Injective ⇑(algebraMap R A)
      h : IsIntegrallyClosedIn R A
      x : A
      hx : Membership.mem (integralClosure R A) x
      ⊢ Membership.mem Bot.bot x
    -/
    rw [Algebra.mem_bot, Set.mem_range]
    /-
      case mpr
      R : Type u_1
      A : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      hRA : Function.Injective ⇑(algebraMap R A)
      h : IsIntegrallyClosedIn R A
      x : A
      hx : Membership.mem (integralClosure R A) x
      ⊢ Exists fun y => Eq ((algebraMap R A) y) x
    -/
    exact isIntegral_iff.mp hx
    /-
      🎉 no goals
    -/


@[simp]
theorem integralClosure_eq_bot [IsIntegrallyClosedIn R A] [NoZeroSMulDivisors R A] [Nontrivial A] :
    integralClosure R A = ⊥ :=
  (integralClosure_eq_bot_iff A (NoZeroSMulDivisors.algebraMap_injective _ _)).mpr ‹_›


/-- If `R` is the integral closure of `S` in `A`, then it is integrally closed in `A`. -/
lemma of_isIntegralClosure [Algebra R B] [Algebra A B] [IsScalarTower R A B]
    [IsIntegralClosure A R B] :
    IsIntegrallyClosedIn A B :=
  have : Algebra.IsIntegral R A := IsIntegralClosure.isIntegral_algebra R B
  IsIntegralClosure.tower_top (R := R)


lemma _root_.IsIntegralClosure.of_isIntegrallyClosedIn
    [Algebra R B] [Algebra A B] [IsScalarTower R A B]
    [IsIntegrallyClosedIn A B] [Algebra.IsIntegral R A] :
    IsIntegralClosure A R B := by
  refine ⟨IsIntegralClosure.algebraMap_injective _ A _, fun {x} ↦
    ⟨fun hx ↦ IsIntegralClosure.isIntegral_iff.mp (IsIntegral.tower_top (A := A) hx), ?_⟩⟩
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    B : Type u_3
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsIntegrallyClosedIn A B
    inst✝ : Algebra.IsIntegral R A
    x : B
    ⊢ (Exists fun y => Eq ((algebraMap A B) y) x) → IsIntegral R x
  -/
  rintro ⟨y, rfl⟩
  /-
    case intro
    R : Type u_1
    A : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    B : Type u_3
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsIntegrallyClosedIn A B
    inst✝ : Algebra.IsIntegral R A
    y : A
    ⊢ IsIntegral R ((algebraMap A B) y)
  -/
  exact IsIntegral.map (IsScalarTower.toAlgHom A A B) (Algebra.IsIntegral.isIntegral y)
  /-
    🎉 no goals
  -/


/-- Note that this is not a duplicate instance, since `IsIntegrallyClosed R` is instead defined
as `IsIntegrallyClosed R R (FractionRing R)`. -/
instance [iic : IsIntegrallyClosed R] : IsIntegralClosure R R K :=
  (isIntegrallyClosed_iff_isIntegralClosure K).mp iic


theorem algebraMap_eq_of_integral [IsIntegrallyClosed R] {x : K} :
    IsIntegral R x → ∃ y : R, algebraMap R K y = x :=
  IsIntegralClosure.isIntegral_iff.mp


theorem isIntegral_iff [IsIntegrallyClosed R] {x : K} :
    IsIntegral R x ↔ ∃ y : R, algebraMap R K y = x :=
  IsIntegrallyClosedIn.isIntegral_iff


theorem exists_algebraMap_eq_of_isIntegral_pow [IsIntegrallyClosed R] {x : K} {n : ℕ} (hn : 0 < n)
    (hx : IsIntegral R <| x ^ n) : ∃ y : R, algebraMap R K y = x :=
  IsIntegrallyClosedIn.exists_algebraMap_eq_of_isIntegral_pow hn hx


theorem exists_algebraMap_eq_of_pow_mem_subalgebra {K : Type*} [CommRing K] [Algebra R K]
    {S : Subalgebra R K} [IsIntegrallyClosed S] [IsFractionRing S K] {x : K} {n : ℕ} (hn : 0 < n)
    (hx : x ^ n ∈ S) : ∃ y : S, algebraMap S K y = x :=
  IsIntegrallyClosedIn.exists_algebraMap_eq_of_pow_mem_subalgebra hn hx


instance _root_.IsIntegralClosure.of_isIntegrallyClosed [IsIntegrallyClosed R]
    [Algebra S R] [Algebra S K] [IsScalarTower S R K] [Algebra.IsIntegral S R] :
    IsIntegralClosure R S K :=
  IsIntegralClosure.of_isIntegrallyClosedIn


theorem integralClosure_eq_bot_iff : integralClosure R K = ⊥ ↔ IsIntegrallyClosed R :=
  (IsIntegrallyClosedIn.integralClosure_eq_bot_iff _ (IsFractionRing.injective _ _)).trans
    (isIntegrallyClosed_iff_isIntegrallyClosedIn _).symm


@[simp]
theorem pow_dvd_pow_iff [IsDomain R] [IsIntegrallyClosed R]
    {n : ℕ} (hn : n ≠ 0) {a b : R} : a ^ n ∣ b ^ n ↔ a ∣ b := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsIntegrallyClosed R
    n : Nat
    hn : Ne n 0
    a b : R
    ⊢ Iff (Dvd.dvd (HPow.hPow a n) (HPow.hPow b n)) (Dvd.dvd a b)
  -/
  refine ⟨fun ⟨x, hx⟩ ↦ ?_, fun h ↦ pow_dvd_pow_of_dvd h n⟩
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsIntegrallyClosed R
    n : Nat
    hn : Ne n 0
    a b : R
    x✝ : Dvd.dvd (HPow.hPow a n) (HPow.hPow b n)
    x : R
    hx : Eq (HPow.hPow b n) (HMul.hMul (HPow.hPow a n) x)
    ⊢ Dvd.dvd a b
  -/
  by_cases ha : a = 0
    /-
      case pos
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : IsIntegrallyClosed R
      n : Nat
      hn : Ne n 0
      a b : R
      x✝ : Dvd.dvd (HPow.hPow a n) (HPow.hPow b n)
      x : R
      hx : Eq (HPow.hPow b n) (HMul.hMul (HPow.hPow a n) x)
      ha : Eq a 0
      ⊢ Dvd.dvd a b
    -/
  · simpa [ha, hn] using hx
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsIntegrallyClosed R
    n : Nat
    hn : Ne n 0
    a b : R
    x✝ : Dvd.dvd (HPow.hPow a n) (HPow.hPow b n)
    x : R
    hx : Eq (HPow.hPow b n) (HMul.hMul (HPow.hPow a n) x)
    ha : Not (Eq a 0)
    ⊢ Dvd.dvd a b
  -/
  let K := FractionRing R
  replace ha : algebraMap R K a ≠ 0 := fun h ↦
    ha <| (injective_iff_map_eq_zero _).1 (IsFractionRing.injective R K) _ h
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsIntegrallyClosed R
    n : Nat
    hn : Ne n 0
    a b : R
    x✝ : Dvd.dvd (HPow.hPow a n) (HPow.hPow b n)
    x : R
    hx : Eq (HPow.hPow b n) (HMul.hMul (HPow.hPow a n) x)
    K : Type u_1 := FractionRing R
    ha : Ne ((algebraMap R K) a) 0
    ⊢ Dvd.dvd a b
  -/
  let y := (algebraMap R K b) / (algebraMap R K a)
  have hy : IsIntegral R y := by
    refine ⟨X ^ n - C x, monic_X_pow_sub_C _ hn, ?_⟩
    simp only [y, map_pow, eval₂_sub, eval₂_X_pow, div_pow, eval₂_pow', eval₂_C]
    replace hx := congr_arg (algebraMap R K) hx
    rw [map_pow] at hx
    field_simp [hx, ha]
  /-
    case neg
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsIntegrallyClosed R
    n : Nat
    hn : Ne n 0
    a b : R
    x✝ : Dvd.dvd (HPow.hPow a n) (HPow.hPow b n)
    x : R
    hx : Eq (HPow.hPow b n) (HMul.hMul (HPow.hPow a n) x)
    K : Type u_1 := FractionRing R
    ha : Ne ((algebraMap R K) a) 0
    y : K := HDiv.hDiv ((algebraMap R K) b) ((algebraMap R K) a)
    hy : IsIntegral R y
    ⊢ Dvd.dvd a b
  -/
  obtain ⟨k, hk⟩ := algebraMap_eq_of_integral hy
  /-
    case neg.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsIntegrallyClosed R
    n : Nat
    hn : Ne n 0
    a b : R
    x✝ : Dvd.dvd (HPow.hPow a n) (HPow.hPow b n)
    x : R
    hx : Eq (HPow.hPow b n) (HMul.hMul (HPow.hPow a n) x)
    K : Type u_1 := FractionRing R
    ha : Ne ((algebraMap R K) a) 0
    y : K := HDiv.hDiv ((algebraMap R K) b) ((algebraMap R K) a)
    hy : IsIntegral R y
    k : R
    hk : Eq ((algebraMap R K) k) y
    ⊢ Dvd.dvd a b
  -/
  refine ⟨k, IsFractionRing.injective R K ?_⟩
  /-
    case neg.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsIntegrallyClosed R
    n : Nat
    hn : Ne n 0
    a b : R
    x✝ : Dvd.dvd (HPow.hPow a n) (HPow.hPow b n)
    x : R
    hx : Eq (HPow.hPow b n) (HMul.hMul (HPow.hPow a n) x)
    K : Type u_1 := FractionRing R
    ha : Ne ((algebraMap R K) a) 0
    y : K := HDiv.hDiv ((algebraMap R K) b) ((algebraMap R K) a)
    hy : IsIntegral R y
    k : R
    hk : Eq ((algebraMap R K) k) y
    ⊢ Eq ((algebraMap R K) b) ((algebraMap R K) (HMul.hMul a k))
  -/
  rw [map_mul, hk, mul_div_cancel₀ _ ha]
  /-
    🎉 no goals
  -/


/-- This is almost a duplicate of `IsIntegrallyClosedIn.integralClosure_eq_bot`,
except the `NoZeroSMulDivisors` hypothesis isn't inferred automatically from `IsFractionRing`. -/
@[simp]
theorem integralClosure_eq_bot [IsIntegrallyClosed R] : integralClosure R K = ⊥ :=
  (integralClosure_eq_bot_iff K).mpr ‹_›


theorem isIntegrallyClosedOfFiniteExtension [IsDomain R] [FiniteDimensional K L] :
    IsIntegrallyClosed (integralClosure R L) :=
  letI : IsFractionRing (integralClosure R L) L := isFractionRing_of_finite_extension K L
  (integralClosure_eq_bot_iff L).mp integralClosure_idem


lemma isIntegrallyClosed_of_isLocalization [IsIntegrallyClosed R] [IsDomain R] (M : Submonoid R)
    (hM : M ≤ R⁰) [IsLocalization M S] : IsIntegrallyClosed S := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDomain R
    M : Submonoid R
    hM : LE.le M (nonZeroDivisors R)
    inst✝ : IsLocalization M S
    ⊢ IsIntegrallyClosed S
  -/
  let K := FractionRing R
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDomain R
    M : Submonoid R
    hM : LE.le M (nonZeroDivisors R)
    inst✝ : IsLocalization M S
    K : Type u_1 := FractionRing R
    ⊢ IsIntegrallyClosed S
  -/
  let g : S →+* K := IsLocalization.map _ (T := R⁰) (RingHom.id R) hM
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDomain R
    M : Submonoid R
    hM : LE.le M (nonZeroDivisors R)
    inst✝ : IsLocalization M S
    K : Type u_1 := FractionRing R
    g : RingHom S K := IsLocalization.map K (RingHom.id R) hM
    ⊢ IsIntegrallyClosed S
  -/
  letI := g.toAlgebra
  have : IsScalarTower R S K := IsScalarTower.of_algebraMap_eq'
    (by rw [RingHom.algebraMap_toAlgebra, IsLocalization.map_comp, RingHomCompTriple.comp_eq])
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDomain R
    M : Submonoid R
    hM : LE.le M (nonZeroDivisors R)
    inst✝ : IsLocalization M S
    K : Type u_1 := FractionRing R
    g : RingHom S K := IsLocalization.map K (RingHom.id R) hM
    this✝ : Algebra S K := g.toAlgebra
    this : IsScalarTower R S K
    ⊢ IsIntegrallyClosed S
  -/
  have := IsFractionRing.isFractionRing_of_isDomain_of_isLocalization M S K
  refine (isIntegrallyClosed_iff_isIntegralClosure (K := K)).mpr
    ⟨IsFractionRing.injective _ _, fun {x} ↦ ⟨?_, fun e ↦ e.choose_spec ▸ isIntegral_algebraMap⟩⟩
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDomain R
    M : Submonoid R
    hM : LE.le M (nonZeroDivisors R)
    inst✝ : IsLocalization M S
    K : Type u_1 := FractionRing R
    g : RingHom S K := IsLocalization.map K (RingHom.id R) hM
    this✝¹ : Algebra S K := g.toAlgebra
    this✝ : IsScalarTower R S K
    this : IsFractionRing S K
    x : K
    ⊢ IsIntegral S x → Exists fun y => Eq ((algebraMap S K) y) x
  -/
  intro hx
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDomain R
    M : Submonoid R
    hM : LE.le M (nonZeroDivisors R)
    inst✝ : IsLocalization M S
    K : Type u_1 := FractionRing R
    g : RingHom S K := IsLocalization.map K (RingHom.id R) hM
    this✝¹ : Algebra S K := g.toAlgebra
    this✝ : IsScalarTower R S K
    this : IsFractionRing S K
    x : K
    hx : IsIntegral S x
    ⊢ Exists fun y => Eq ((algebraMap S K) y) x
  -/
  obtain ⟨⟨y, y_mem⟩, hy⟩ := hx.exists_multiple_integral_of_isLocalization M _
  /-
    case intro.mk
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDomain R
    M : Submonoid R
    hM : LE.le M (nonZeroDivisors R)
    inst✝ : IsLocalization M S
    K : Type u_1 := FractionRing R
    g : RingHom S K := IsLocalization.map K (RingHom.id R) hM
    this✝¹ : Algebra S K := g.toAlgebra
    this✝ : IsScalarTower R S K
    this : IsFractionRing S K
    x : K
    hx : IsIntegral S x
    y : R
    y_mem : Membership.mem M y
    hy : IsIntegral R (HSMul.hSMul ⟨y, y_mem⟩ x)
    ⊢ Exists fun y => Eq ((algebraMap S K) y) x
  -/
  obtain ⟨z, hz⟩ := (isIntegrallyClosed_iff _).mp ‹_› hy
  /-
    case intro.mk.intro
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDomain R
    M : Submonoid R
    hM : LE.le M (nonZeroDivisors R)
    inst✝ : IsLocalization M S
    K : Type u_1 := FractionRing R
    g : RingHom S K := IsLocalization.map K (RingHom.id R) hM
    this✝¹ : Algebra S K := g.toAlgebra
    this✝ : IsScalarTower R S K
    this : IsFractionRing S K
    x : K
    hx : IsIntegral S x
    y : R
    y_mem : Membership.mem M y
    hy : IsIntegral R (HSMul.hSMul ⟨y, y_mem⟩ x)
    z : R
    hz : Eq ((algebraMap R K) z) (HSMul.hSMul ⟨y, y_mem⟩ x)
    ⊢ Exists fun y => Eq ((algebraMap S K) y) x
  -/
  refine ⟨IsLocalization.mk' S z ⟨y, y_mem⟩, (IsLocalization.lift_mk'_spec _ _ _ _).mpr ?_⟩
  /-
    case intro.mk.intro
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDomain R
    M : Submonoid R
    hM : LE.le M (nonZeroDivisors R)
    inst✝ : IsLocalization M S
    K : Type u_1 := FractionRing R
    g : RingHom S K := IsLocalization.map K (RingHom.id R) hM
    this✝¹ : Algebra S K := g.toAlgebra
    this✝ : IsScalarTower R S K
    this : IsFractionRing S K
    x : K
    hx : IsIntegral S x
    y : R
    y_mem : Membership.mem M y
    hy : IsIntegral R (HSMul.hSMul ⟨y, y_mem⟩ x)
    z : R
    hz : Eq ((algebraMap R K) z) (HSMul.hSMul ⟨y, y_mem⟩ x)
    ⊢ Eq (((algebraMap R K).comp (RingHom.id R)) z) (HMul.hMul (((algebraMap R K). …
  -/
  rw [RingHom.comp_id, hz, ← Algebra.smul_def, Submonoid.mk_smul]
  /-
    🎉 no goals
  -/


/-- Any field is integral closed. -/
/- Although `infer_instance` can find this if you import Mathlib, in this file they have not been
  proven yet. However, it is used to prove a fundamental property of `IsIntegrallyClosed`,
  and it is not desirable to involve more content from other files. -/
instance Field.instIsIntegrallyClosed (K : Type*) [Field K] : IsIntegrallyClosed K :=
  (isIntegrallyClosed_iff K).mpr fun {x} _ ↦ ⟨x, rfl⟩

