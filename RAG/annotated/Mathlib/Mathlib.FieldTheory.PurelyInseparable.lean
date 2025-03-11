/-- Typeclass for purely inseparable field extensions: an algebraic extension `E / F` is purely
inseparable if and only if the minimal polynomial of every element of `E ∖ F` is not separable.

We define this for general (commutative) rings and only assume `F` and `E` are fields
if this is needed for a proof. -/
class IsPurelyInseparable : Prop where
  isIntegral : Algebra.IsIntegral F E
  inseparable' (x : E) : IsSeparable F x → x ∈ (algebraMap F E).range


variable {E} in
theorem IsPurelyInseparable.isIntegral' [IsPurelyInseparable F E] (x : E) : IsIntegral F x :=
  Algebra.IsIntegral.isIntegral _


theorem IsPurelyInseparable.isAlgebraic [Nontrivial F] [IsPurelyInseparable F E] :
    Algebra.IsAlgebraic F E := inferInstance


theorem IsPurelyInseparable.inseparable [IsPurelyInseparable F E] :
    ∀ x : E, IsSeparable F x → x ∈ (algebraMap F E).range :=
  IsPurelyInseparable.inseparable'


theorem isPurelyInseparable_iff : IsPurelyInseparable F E ↔ ∀ x : E,
    IsIntegral F x ∧ (IsSeparable F x → x ∈ (algebraMap F E).range) :=
  ⟨fun h x ↦ ⟨h.isIntegral' _ x, h.inseparable' x⟩, fun h ↦ ⟨⟨fun x ↦ (h x).1⟩, fun x ↦ (h x).2⟩⟩


/-- Transfer `IsPurelyInseparable` across an `AlgEquiv`. -/
theorem AlgEquiv.isPurelyInseparable (e : K ≃ₐ[F] E) [IsPurelyInseparable F K] :
    IsPurelyInseparable F E := by
  refine ⟨⟨fun _ ↦ by rw [← isIntegral_algEquiv e.symm]; exact IsPurelyInseparable.isIntegral' F _⟩,
    fun x h ↦ ?_⟩
  /-
    F : Type u
    E : Type v
    inst✝⁵ : CommRing F
    inst✝⁴ : Ring E
    inst✝³ : Algebra F E
    K : Type w
    inst✝² : Ring K
    inst✝¹ : Algebra F K
    e : AlgEquiv F K E
    inst✝ : IsPurelyInseparable F K
    x : E
    h : IsSeparable F x
    ⊢ Membership.mem (algebraMap F E).range x
  -/
  rw [IsSeparable, ← minpoly.algEquiv_eq e.symm] at h
  /-
    F : Type u
    E : Type v
    inst✝⁵ : CommRing F
    inst✝⁴ : Ring E
    inst✝³ : Algebra F E
    K : Type w
    inst✝² : Ring K
    inst✝¹ : Algebra F K
    e : AlgEquiv F K E
    inst✝ : IsPurelyInseparable F K
    x : E
    h : (minpoly F (e.symm x)).Separable
    ⊢ Membership.mem (algebraMap F E).range x
  -/
  simpa only [RingHom.mem_range, algebraMap_eq_apply] using IsPurelyInseparable.inseparable F _ h
  /-
    🎉 no goals
  -/


theorem AlgEquiv.isPurelyInseparable_iff (e : K ≃ₐ[F] E) :
    IsPurelyInseparable F K ↔ IsPurelyInseparable F E :=
  ⟨fun _ ↦ e.isPurelyInseparable, fun _ ↦ e.symm.isPurelyInseparable⟩


/-- If `E / F` is an algebraic extension, `F` is separably closed,
then `E / F` is purely inseparable. -/
instance Algebra.IsAlgebraic.isPurelyInseparable_of_isSepClosed
    {F : Type u} {E : Type v} [Field F] [Ring E] [IsDomain E] [Algebra F E]
    [Algebra.IsAlgebraic F E]
    [IsSepClosed F] : IsPurelyInseparable F E :=
  ⟨inferInstance, fun x h ↦ minpoly.mem_range_of_degree_eq_one F x <|
    IsSepClosed.degree_eq_one_of_irreducible F (minpoly.irreducible
      (Algebra.IsIntegral.isIntegral _)) h⟩


/-- If `E / F` is both purely inseparable and separable, then `algebraMap F E` is surjective. -/
theorem IsPurelyInseparable.surjective_algebraMap_of_isSeparable
    [IsPurelyInseparable F E] [Algebra.IsSeparable F E] : Function.Surjective (algebraMap F E) :=
  fun x ↦ IsPurelyInseparable.inseparable F x (Algebra.IsSeparable.isSeparable F x)


/-- If `E / F` is both purely inseparable and separable, then `algebraMap F E` is bijective. -/
theorem IsPurelyInseparable.bijective_algebraMap_of_isSeparable
    [Nontrivial E] [NoZeroSMulDivisors F E]
    [IsPurelyInseparable F E] [Algebra.IsSeparable F E] : Function.Bijective (algebraMap F E) :=
  ⟨NoZeroSMulDivisors.algebraMap_injective F E, surjective_algebraMap_of_isSeparable F E⟩


variable {F E} in
/-- If a subalgebra of `E / F` is both purely inseparable and separable, then it is equal
to `F`. -/
theorem Subalgebra.eq_bot_of_isPurelyInseparable_of_isSeparable (L : Subalgebra F E)
    [IsPurelyInseparable F L] [Algebra.IsSeparable F L] : L = ⊥ := bot_unique fun x hx ↦ by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : CommRing F
    inst✝³ : Ring E
    inst✝² : Algebra F E
    L : Subalgebra F E
    inst✝¹ : IsPurelyInseparable F (Subtype fun x => Membership.mem L x)
    inst✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
    x : E
    hx : Membership.mem L x
    ⊢ Membership.mem Bot.bot x
  -/
  obtain ⟨y, hy⟩ := IsPurelyInseparable.surjective_algebraMap_of_isSeparable F L ⟨x, hx⟩
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁴ : CommRing F
    inst✝³ : Ring E
    inst✝² : Algebra F E
    L : Subalgebra F E
    inst✝¹ : IsPurelyInseparable F (Subtype fun x => Membership.mem L x)
    inst✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
    x : E
    hx : Membership.mem L x
    y : F
    hy : Eq ((algebraMap F (Subtype fun x => Membership.mem L x)) y) ⟨x, hx⟩
    ⊢ Membership.mem Bot.bot x
  -/
  exact ⟨y, congr_arg (Subalgebra.val _) hy⟩
  /-
    🎉 no goals
  -/


/-- If an intermediate field of `E / F` is both purely inseparable and separable, then it is equal
to `F`. -/
theorem IntermediateField.eq_bot_of_isPurelyInseparable_of_isSeparable
    {F : Type u} {E : Type v} [Field F] [Field E] [Algebra F E] (L : IntermediateField F E)
    [IsPurelyInseparable F L] [Algebra.IsSeparable F L] : L = ⊥ := bot_unique fun x hx ↦ by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    L : IntermediateField F E
    inst✝¹ : IsPurelyInseparable F (Subtype fun x => Membership.mem L x)
    inst✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
    x : E
    hx : Membership.mem L x
    ⊢ Membership.mem Bot.bot x
  -/
  obtain ⟨y, hy⟩ := IsPurelyInseparable.surjective_algebraMap_of_isSeparable F L ⟨x, hx⟩
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    L : IntermediateField F E
    inst✝¹ : IsPurelyInseparable F (Subtype fun x => Membership.mem L x)
    inst✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
    x : E
    hx : Membership.mem L x
    y : F
    hy : Eq ((algebraMap F (Subtype fun x => Membership.mem L x)) y) ⟨x, hx⟩
    ⊢ Membership.mem Bot.bot x
  -/
  exact ⟨y, congr_arg (algebraMap L E) hy⟩
  /-
    🎉 no goals
  -/


/-- If `E / F` is purely inseparable, then the separable closure of `F` in `E` is
equal to `F`. -/
theorem separableClosure.eq_bot_of_isPurelyInseparable
    (F : Type u) (E : Type v) [Field F] [Field E] [Algebra F E] [IsPurelyInseparable F E] :
    separableClosure F E = ⊥ :=
  bot_unique fun x h ↦ IsPurelyInseparable.inseparable F x (mem_separableClosure_iff.1 h)


/-- If `E / F` is an algebraic extension, then the separable closure of `F` in `E` is
equal to `F` if and only if `E / F` is purely inseparable. -/
theorem separableClosure.eq_bot_iff
    {F : Type u} {E : Type v} [Field F] [Field E] [Algebra F E] [Algebra.IsAlgebraic F E] :
    separableClosure F E = ⊥ ↔ IsPurelyInseparable F E :=
  ⟨fun h ↦ isPurelyInseparable_iff.2 fun x ↦ ⟨Algebra.IsIntegral.isIntegral x, fun hs ↦ by
    /-
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : Algebra.IsAlgebraic F E
      h : Eq (separableClosure F E) Bot.bot
      x : E
      hs : IsSeparable F x
      ⊢ Membership.mem (algebraMap F E).range x
    -/
    simpa only [h] using mem_separableClosure_iff.2 hs⟩, fun _ ↦ eq_bot_of_isPurelyInseparable F E⟩
    /-
      🎉 no goals
    -/


instance isPurelyInseparable_self : IsPurelyInseparable F F :=
  ⟨inferInstance, fun x _ ↦ ⟨x, rfl⟩⟩


/-- A field extension `E / F` of exponential characteristic `q` is purely inseparable
if and only if for every element `x` of `E`, there exists a natural number `n` such that
`x ^ (q ^ n)` is contained in `F`. -/
@[stacks 09HE]
theorem isPurelyInseparable_iff_pow_mem :
    IsPurelyInseparable F E ↔ ∀ x : E, ∃ n : ℕ, x ^ q ^ n ∈ (algebraMap F E).range := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Ring E
    inst✝² : IsDomain E
    inst✝¹ : Algebra F E
    q : Nat
    inst✝ : ExpChar F q
    ⊢ Iff (IsPurelyInseparable F E) (∀ (x : E), Exists fun n => Membership.mem (al …
  -/
  rw [isPurelyInseparable_iff]
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Ring E
    inst✝² : IsDomain E
    inst✝¹ : Algebra F E
    q : Nat
    inst✝ : ExpChar F q
    ⊢ Iff (∀ (x : E), And (IsIntegral F x) (IsSeparable F x → Membership.mem (alge …
  -/
  refine ⟨fun h x ↦ ?_, fun h x ↦ ?_⟩
    /-
      case refine_1
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Ring E
      inst✝² : IsDomain E
      inst✝¹ : Algebra F E
      q : Nat
      inst✝ : ExpChar F q
      h : ∀ (x : E), And (IsIntegral F x) (IsSeparable F x → Membership.mem (algebra …
      x : E
      ⊢ Exists fun n => Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPo …
    -/
  · obtain ⟨g, h1, n, h2⟩ := (minpoly.irreducible (h x).1).hasSeparableContraction q
    exact ⟨n, (h _).2 <| h1.of_dvd <| minpoly.dvd F _ <| by
      simpa only [expand_aeval, minpoly.aeval] using congr_arg (aeval x) h2⟩
  /-
    case refine_2
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Ring E
    inst✝² : IsDomain E
    inst✝¹ : Algebra F E
    q : Nat
    inst✝ : ExpChar F q
    h : ∀ (x : E), Exists fun n => Membership.mem (algebraMap F E).range (HPow.hPo …
    x : E
    ⊢ And (IsIntegral F x) (IsSeparable F x → Membership.mem (algebraMap F E).rang …
  -/
  have hdeg := (minpoly.natSepDegree_eq_one_iff_pow_mem q).2 (h x)
  have halg : IsIntegral F x := by_contra fun h' ↦ by
    simp only [minpoly.eq_zero h', natSepDegree_zero, zero_ne_one] at hdeg
  /-
    case refine_2
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Ring E
    inst✝² : IsDomain E
    inst✝¹ : Algebra F E
    q : Nat
    inst✝ : ExpChar F q
    h : ∀ (x : E), Exists fun n => Membership.mem (algebraMap F E).range (HPow.hPo …
    x : E
    hdeg : Eq (minpoly F x).natSepDegree 1
    halg : IsIntegral F x
    ⊢ And (IsIntegral F x) (IsSeparable F x → Membership.mem (algebraMap F E).rang …
  -/
  refine ⟨halg, fun hsep ↦ ?_⟩
  /-
    case refine_2
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Ring E
    inst✝² : IsDomain E
    inst✝¹ : Algebra F E
    q : Nat
    inst✝ : ExpChar F q
    h : ∀ (x : E), Exists fun n => Membership.mem (algebraMap F E).range (HPow.hPo …
    x : E
    hdeg : Eq (minpoly F x).natSepDegree 1
    halg : IsIntegral F x
    hsep : IsSeparable F x
    ⊢ Membership.mem (algebraMap F E).range x
  -/
  rwa [hsep.natSepDegree_eq_natDegree, minpoly.natDegree_eq_one_iff] at hdeg
  /-
    🎉 no goals
  -/


theorem IsPurelyInseparable.pow_mem [IsPurelyInseparable F E] :
    ∃ n : ℕ, x ^ q ^ n ∈ (algebraMap F E).range :=
  (isPurelyInseparable_iff_pow_mem F q).1 ‹_› x


/-- The relative perfect closure of `F` in `E`, consists of the elements `x` of `E` such that there
exists a natural number `n` such that `x ^ (ringExpChar F) ^ n` is contained in `F`, where
`ringExpChar F` is the exponential characteristic of `F`. It is also the maximal purely inseparable
subextension of `E / F` (`le_perfectClosure_iff`). -/
@[stacks 09HH]
def perfectClosure : IntermediateField F E where
  carrier := {x : E | ∃ n : ℕ, x ^ (ringExpChar F) ^ n ∈ (algebraMap F E).range}
  add_mem' := by
    /-
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      ⊢ ∀ {a b : E}, Membership.mem { carrier := setOf fun x => Exists fun n => Memb …
    -/
    rintro x y ⟨n, hx⟩ ⟨m, hy⟩
    /-
      case intro.intro
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      x y : E
      n : Nat
      hx : Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPow (ringExpCha …
      m : Nat
      hy : Membership.mem (algebraMap F E).range (HPow.hPow y (HPow.hPow (ringExpCha …
      ⊢ Membership.mem { carrier := setOf fun x => Exists fun n => Membership.mem (a …
    -/
    use n + m
    /-
      case h
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      x y : E
      n : Nat
      hx : Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPow (ringExpCha …
      m : Nat
      hy : Membership.mem (algebraMap F E).range (HPow.hPow y (HPow.hPow (ringExpCha …
      ⊢ Membership.mem (algebraMap F E).range (HPow.hPow (HAdd.hAdd x y) (HPow.hPow  …
    -/
    have := expChar_of_injective_algebraMap (algebraMap F E).injective (ringExpChar F)
    /-
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      ⊢ ∀ {a b : E}, Membership.mem (setOf fun x => Exists fun n => Membership.mem ( …
    -/
    /-
      case h
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      x y : E
      n : Nat
      hx : Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPow (ringExpCha …
      m : Nat
      hy : Membership.mem (algebraMap F E).range (HPow.hPow y (HPow.hPow (ringExpCha …
      this : ExpChar E (ringExpChar F)
      ⊢ Membership.mem (algebraMap F E).range (HPow.hPow (HAdd.hAdd x y) (HPow.hPow  …
    -/
    /-
      case intro.intro
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      x y : E
      n : Nat
      hx : Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPow (ringExpCha …
      m : Nat
      hy : Membership.mem (algebraMap F E).range (HPow.hPow y (HPow.hPow (ringExpCha …
      ⊢ Membership.mem (setOf fun x => Exists fun n => Membership.mem (algebraMap F  …
    -/
    rw [add_pow_expChar_pow, pow_add, pow_mul, mul_comm (_ ^ n), pow_mul]
    /-
      case h
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      x y : E
      n : Nat
      hx : Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPow (ringExpCha …
      m : Nat
      hy : Membership.mem (algebraMap F E).range (HPow.hPow y (HPow.hPow (ringExpCha …
      ⊢ Membership.mem (algebraMap F E).range (HPow.hPow (HMul.hMul x y) (HPow.hPow  …
    -/
    /-
      case h
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      x y : E
      n : Nat
      hx : Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPow (ringExpCha …
      m : Nat
      hy : Membership.mem (algebraMap F E).range (HPow.hPow y (HPow.hPow (ringExpCha …
      this : ExpChar E (ringExpChar F)
      ⊢ Membership.mem (algebraMap F E).range (HAdd.hAdd (HPow.hPow (HPow.hPow x (HP …
    -/
    /-
      case h
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      x y : E
      n : Nat
      hx : Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPow (ringExpCha …
      m : Nat
      hy : Membership.mem (algebraMap F E).range (HPow.hPow y (HPow.hPow (ringExpCha …
      ⊢ Membership.mem (algebraMap F E).range (HMul.hMul (HPow.hPow (HPow.hPow x (HP …
    -/
    exact add_mem (pow_mem hx _) (pow_mem hy _)
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  mul_mem' := by
    rintro x y ⟨n, hx⟩ ⟨m, hy⟩
    use n + m
    rw [mul_pow, pow_add, pow_mul, mul_comm (_ ^ n), pow_mul]
    exact mul_mem (pow_mem hx _) (pow_mem hy _)
  inv_mem' := by
    /-
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      ⊢ ∀ (x : E), Membership.mem { carrier := setOf fun x => Exists fun n => Member …
    -/
    rintro x ⟨n, hx⟩
    /-
      case intro
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      x : E
      n : Nat
      hx : Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPow (ringExpCha …
      ⊢ Membership.mem { carrier := setOf fun x => Exists fun n => Membership.mem (a …
    -/
                                    /-
                                      F : Type u
                                      E : Type v
                                      inst✝⁴ : Field F
                                      inst✝³ : Field E
                                      inst✝² : Algebra F E
                                      K : Type w
                                      inst✝¹ : Field K
                                      inst✝ : Algebra F K
                                      x : F
                                      ⊢ Membership.mem (algebraMap F E).range (HPow.hPow ((algebraMap F E) x) (HPow. …
                                    -/
    use n; rw [inv_pow]
                                                            /-
                                                              🎉 no goals
                                                            -/
    /-
      case h
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      x : E
      n : Nat
      hx : Membership.mem (algebraMap F E).range (HPow.hPow x (HPow.hPow (ringExpCha …
      ⊢ Membership.mem (algebraMap F E).range (Inv.inv (HPow.hPow x (HPow.hPow (ring …
    -/
    apply inv_mem (id hx : _ ∈ (⊥ : IntermediateField F E))
    /-
      🎉 no goals
    -/
  algebraMap_mem' := fun x ↦ ⟨0, by rw [pow_zero, pow_one]; exact ⟨x, rfl⟩⟩


theorem mem_perfectClosure_iff {x : E} :
    x ∈ perfectClosure F E ↔ ∃ n : ℕ, x ^ (ringExpChar F) ^ n ∈ (algebraMap F E).range := Iff.rfl


theorem mem_perfectClosure_iff_pow_mem (q : ℕ) [ExpChar F q] {x : E} :
    x ∈ perfectClosure F E ↔ ∃ n : ℕ, x ^ q ^ n ∈ (algebraMap F E).range := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    q : Nat
    inst✝ : ExpChar F q
    x : E
    ⊢ Iff (Membership.mem (perfectClosure F E) x) (Exists fun n => Membership.mem  …
  -/
  rw [mem_perfectClosure_iff, ringExpChar.eq F q]
  /-
    🎉 no goals
  -/


/-- An element is contained in the relative perfect closure if and only if its minimal polynomial
has separable degree one. -/
theorem mem_perfectClosure_iff_natSepDegree_eq_one {x : E} :
    x ∈ perfectClosure F E ↔ (minpoly F x).natSepDegree = 1 := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x : E
    ⊢ Iff (Membership.mem (perfectClosure F E) x) (Eq (minpoly F x).natSepDegree 1)
  -/
  rw [mem_perfectClosure_iff, minpoly.natSepDegree_eq_one_iff_pow_mem (ringExpChar F)]
  /-
    🎉 no goals
  -/


/-- A field extension `E / F` is purely inseparable if and only if the relative perfect closure of
`F` in `E` is equal to `E`. -/
theorem isPurelyInseparable_iff_perfectClosure_eq_top :
    IsPurelyInseparable F E ↔ perfectClosure F E = ⊤ := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Iff (IsPurelyInseparable F E) (Eq (perfectClosure F E) Top.top)
  -/
  rw [isPurelyInseparable_iff_pow_mem F (ringExpChar F)]
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Iff (∀ (x : E), Exists fun n => Membership.mem (algebraMap F E).range (HPow. …
  -/
  exact ⟨fun H ↦ top_unique fun x _ ↦ H x, fun H _ ↦ H.ge trivial⟩
  /-
    🎉 no goals
  -/


/-- The relative perfect closure of `F` in `E` is purely inseparable over `F`. -/
instance perfectClosure.isPurelyInseparable : IsPurelyInseparable F (perfectClosure F E) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ⊢ IsPurelyInseparable F (Subtype fun x => Membership.mem (perfectClosure F E) x)
  -/
  rw [isPurelyInseparable_iff_pow_mem F (ringExpChar F)]
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ⊢ ∀ (x : Subtype fun x => Membership.mem (perfectClosure F E) x), Exists fun n …
  -/
  exact fun ⟨_, n, y, h⟩ ↦ ⟨n, y, (algebraMap _ E).injective h⟩
  /-
    🎉 no goals
  -/


/-- The relative perfect closure of `F` in `E` is algebraic over `F`. -/
instance perfectClosure.isAlgebraic : Algebra.IsAlgebraic F (perfectClosure F E) :=
  IsPurelyInseparable.isAlgebraic F _


/-- If `E / F` is separable, then the perfect closure of `F` in `E` is equal to `F`. Note that
  the converse is not necessarily true (see https://math.stackexchange.com/a/3009197)
  even when `E / F` is algebraic. -/
theorem perfectClosure.eq_bot_of_isSeparable [Algebra.IsSeparable F E] : perfectClosure F E = ⊥ :=
  haveI := Algebra.isSeparable_tower_bot_of_isSeparable F (perfectClosure F E) E
  eq_bot_of_isPurelyInseparable_of_isSeparable _


/-- An intermediate field of `E / F` is contained in the relative perfect closure of `F` in `E`
if it is purely inseparable over `F`. -/
theorem le_perfectClosure (L : IntermediateField F E) [h : IsPurelyInseparable F L] :
    L ≤ perfectClosure F E := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    L : IntermediateField F E
    h : IsPurelyInseparable F (Subtype fun x => Membership.mem L x)
    ⊢ LE.le L (perfectClosure F E)
  -/
  rw [isPurelyInseparable_iff_pow_mem F (ringExpChar F)] at h
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    L : IntermediateField F E
    h : ∀ (x : Subtype fun x => Membership.mem L x), Exists fun n => Membership.me …
    ⊢ LE.le L (perfectClosure F E)
  -/
  intro x hx
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    L : IntermediateField F E
    h : ∀ (x : Subtype fun x => Membership.mem L x), Exists fun n => Membership.me …
    x : E
    hx : Membership.mem L x
    ⊢ Membership.mem (perfectClosure F E) x
  -/
  obtain ⟨n, y, hy⟩ := h ⟨x, hx⟩
  /-
    case intro.intro
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    L : IntermediateField F E
    h : ∀ (x : Subtype fun x => Membership.mem L x), Exists fun n => Membership.me …
    x : E
    hx : Membership.mem L x
    n : Nat
    y : F
    hy : Eq ((algebraMap F (Subtype fun x => Membership.mem L x)) y) (HPow.hPow ⟨x …
    ⊢ Membership.mem (perfectClosure F E) x
  -/
  exact ⟨n, y, congr_arg (algebraMap L E) hy⟩
  /-
    🎉 no goals
  -/


/-- An intermediate field of `E / F` is contained in the relative perfect closure of `F` in `E`
if and only if it is purely inseparable over `F`. -/
theorem le_perfectClosure_iff (L : IntermediateField F E) :
    L ≤ perfectClosure F E ↔ IsPurelyInseparable F L := by
  refine ⟨fun h ↦ (isPurelyInseparable_iff_pow_mem F (ringExpChar F)).2 fun x ↦ ?_,
    fun _ ↦ le_perfectClosure F E L⟩
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    L : IntermediateField F E
    h : LE.le L (perfectClosure F E)
    x : Subtype fun x => Membership.mem L x
    ⊢ Exists fun n => Membership.mem (algebraMap F (Subtype fun x => Membership.me …
  -/
  obtain ⟨n, y, hy⟩ := h x.2
  /-
    case intro.intro
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    L : IntermediateField F E
    h : LE.le L (perfectClosure F E)
    x : Subtype fun x => Membership.mem L x
    n : Nat
    y : F
    hy : Eq ((algebraMap F E) y) (HPow.hPow (↑x) (HPow.hPow (ringExpChar F) n))
    ⊢ Exists fun n => Membership.mem (algebraMap F (Subtype fun x => Membership.me …
  -/
  exact ⟨n, y, (algebraMap L E).injective hy⟩
  /-
    🎉 no goals
  -/


theorem separableClosure_inf_perfectClosure : separableClosure F E ⊓ perfectClosure F E = ⊥ :=
  haveI := (le_separableClosure_iff F E _).mp (inf_le_left (b := perfectClosure F E))
  haveI := (le_perfectClosure_iff F E _).mp (inf_le_right (a := separableClosure F E))
  eq_bot_of_isPurelyInseparable_of_isSeparable _


/-- If `i` is an `F`-algebra homomorphism from `E` to `K`, then `i x` is contained in
`perfectClosure F K` if and only if `x` is contained in `perfectClosure F E`. -/
theorem map_mem_perfectClosure_iff (i : E →ₐ[F] K) {x : E} :
    i x ∈ perfectClosure F K ↔ x ∈ perfectClosure F E := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    i : AlgHom F E K
    x : E
    ⊢ Iff (Membership.mem (perfectClosure F K) (i x)) (Membership.mem (perfectClos …
  -/
  simp_rw [mem_perfectClosure_iff]
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    i : AlgHom F E K
    x : E
    ⊢ Iff (Exists fun n => Membership.mem (algebraMap F K).range (HPow.hPow (i x)  …
  -/
  refine ⟨fun ⟨n, y, h⟩ ↦ ⟨n, y, ?_⟩, fun ⟨n, y, h⟩ ↦ ⟨n, y, ?_⟩⟩
    /-
      case refine_1
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      i : AlgHom F E K
      x : E
      x✝ : Exists fun n => Membership.mem (algebraMap F K).range (HPow.hPow (i x) (H …
      n : Nat
      y : F
      h : Eq ((algebraMap F K) y) (HPow.hPow (i x) (HPow.hPow (ringExpChar F) n))
      ⊢ Eq ((algebraMap F E) y) (HPow.hPow x (HPow.hPow (ringExpChar F) n))
    -/
  · apply_fun i using i.injective
    /-
      case refine_1
      F : Type u
      E : Type v
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Algebra F E
      K : Type w
      inst✝¹ : Field K
      inst✝ : Algebra F K
      i : AlgHom F E K
      x : E
      x✝ : Exists fun n => Membership.mem (algebraMap F K).range (HPow.hPow (i x) (H …
      n : Nat
      y : F
      h : Eq ((algebraMap F K) y) (HPow.hPow (i x) (HPow.hPow (ringExpChar F) n))
      ⊢ Eq (i ((algebraMap F E) y)) (i (HPow.hPow x (HPow.hPow (ringExpChar F) n)))
    -/
    rwa [AlgHom.commutes, map_pow]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    i : AlgHom F E K
    x : E
    x✝ : Exists fun n => Membership.mem (algebraMap F E).range (HPow.hPow x (HPow. …
    n : Nat
    y : F
    h : Eq ((algebraMap F E) y) (HPow.hPow x (HPow.hPow (ringExpChar F) n))
    ⊢ Eq ((algebraMap F K) y) (HPow.hPow (i x) (HPow.hPow (ringExpChar F) n))
  -/
  simpa only [AlgHom.commutes, map_pow] using congr_arg i h
  /-
    🎉 no goals
  -/


/-- If `i` is an `F`-algebra homomorphism from `E` to `K`, then the preimage of `perfectClosure F K`
under the map `i` is equal to `perfectClosure F E`. -/
theorem perfectClosure.comap_eq_of_algHom (i : E →ₐ[F] K) :
    (perfectClosure F K).comap i = perfectClosure F E := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    i : AlgHom F E K
    ⊢ Eq (IntermediateField.comap i (perfectClosure F K)) (perfectClosure F E)
  -/
  ext x
  /-
    case h
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    i : AlgHom F E K
    x : E
    ⊢ Iff (Membership.mem (IntermediateField.comap i (perfectClosure F K)) x) (Mem …
  -/
  exact map_mem_perfectClosure_iff i
  /-
    🎉 no goals
  -/


/-- If `i` is an `F`-algebra homomorphism from `E` to `K`, then the image of `perfectClosure F E`
under the map `i` is contained in `perfectClosure F K`. -/
theorem perfectClosure.map_le_of_algHom (i : E →ₐ[F] K) :
    (perfectClosure F E).map i ≤ perfectClosure F K :=
  map_le_iff_le_comap.mpr (perfectClosure.comap_eq_of_algHom i).ge


/-- If `i` is an `F`-algebra isomorphism of `E` and `K`, then the image of `perfectClosure F E`
under the map `i` is equal to in `perfectClosure F K`. -/
theorem perfectClosure.map_eq_of_algEquiv (i : E ≃ₐ[F] K) :
    (perfectClosure F E).map i.toAlgHom = perfectClosure F K :=
  (map_le_of_algHom i.toAlgHom).antisymm (fun x hx ↦ ⟨i.symm x,
    (map_mem_perfectClosure_iff i.symm.toAlgHom).2 hx, i.right_inv x⟩)


/-- If `E` and `K` are isomorphic as `F`-algebras, then `perfectClosure F E` and
`perfectClosure F K` are also isomorphic as `F`-algebras. -/
def perfectClosure.algEquivOfAlgEquiv (i : E ≃ₐ[F] K) :
    perfectClosure F E ≃ₐ[F] perfectClosure F K :=
  (intermediateFieldMap i _).trans (equivOfEq (map_eq_of_algEquiv i))


alias AlgEquiv.perfectClosure := perfectClosure.algEquivOfAlgEquiv


/-- If `E` is a perfect field of exponential characteristic `p`, then the (relative) perfect closure
`perfectClosure F E` is perfect. -/
instance perfectClosure.perfectRing (p : ℕ) [ExpChar E p]
    [PerfectRing E p] : PerfectRing (perfectClosure F E) p := .ofSurjective _ p fun x ↦ by
  /-
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    p : Nat
    inst✝¹ : ExpChar E p
    inst✝ : PerfectRing E p
    x : Subtype fun x => Membership.mem (perfectClosure F E) x
    ⊢ Exists fun a => Eq ((frobenius (Subtype fun x => Membership.mem (perfectClos …
  -/
  haveI := RingHom.expChar _ (algebraMap F E).injective p
  /-
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    p : Nat
    inst✝¹ : ExpChar E p
    inst✝ : PerfectRing E p
    x : Subtype fun x => Membership.mem (perfectClosure F E) x
    this : ExpChar F p
    ⊢ Exists fun a => Eq ((frobenius (Subtype fun x => Membership.mem (perfectClos …
  -/
  obtain ⟨x', hx⟩ := surjective_frobenius E p x.1
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    p : Nat
    inst✝¹ : ExpChar E p
    inst✝ : PerfectRing E p
    x : Subtype fun x => Membership.mem (perfectClosure F E) x
    this : ExpChar F p
    x' : E
    hx : Eq ((frobenius E p) x') ↑x
    ⊢ Exists fun a => Eq ((frobenius (Subtype fun x => Membership.mem (perfectClos …
  -/
  obtain ⟨n, y, hy⟩ := (mem_perfectClosure_iff_pow_mem p).1 x.2
  /-
    case intro.intro.intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    p : Nat
    inst✝¹ : ExpChar E p
    inst✝ : PerfectRing E p
    x : Subtype fun x => Membership.mem (perfectClosure F E) x
    this : ExpChar F p
    x' : E
    hx : Eq ((frobenius E p) x') ↑x
    n : Nat
    y : F
    hy : Eq ((algebraMap F E) y) (HPow.hPow (↑x) (HPow.hPow p n))
    ⊢ Exists fun a => Eq ((frobenius (Subtype fun x => Membership.mem (perfectClos …
  -/
  rw [frobenius_def] at hx
  /-
    case intro.intro.intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    p : Nat
    inst✝¹ : ExpChar E p
    inst✝ : PerfectRing E p
    x : Subtype fun x => Membership.mem (perfectClosure F E) x
    this : ExpChar F p
    x' : E
    hx : Eq (HPow.hPow x' p) ↑x
    n : Nat
    y : F
    hy : Eq ((algebraMap F E) y) (HPow.hPow (↑x) (HPow.hPow p n))
    ⊢ Exists fun a => Eq ((frobenius (Subtype fun x => Membership.mem (perfectClos …
  -/
  rw [← hx, ← pow_mul, ← pow_succ'] at hy
  exact ⟨⟨x', (mem_perfectClosure_iff_pow_mem p).2 ⟨n + 1, y, hy⟩⟩, by
    simp_rw [frobenius_def, SubmonoidClass.mk_pow, hx]⟩


/-- If `E` is a perfect field, then the (relative) perfect closure
`perfectClosure F E` is perfect. -/
instance perfectClosure.perfectField [PerfectField E] : PerfectField (perfectClosure F E) :=
  PerfectRing.toPerfectField _ (ringExpChar E)


/-- If `K / E / F` is a field extension tower such that `K / F` is purely inseparable,
then `E / F` is also purely inseparable. -/
theorem IsPurelyInseparable.tower_bot [Algebra E K] [IsScalarTower F E K]
    [IsPurelyInseparable F K] : IsPurelyInseparable F E := by
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F K
    ⊢ IsPurelyInseparable F E
  -/
  refine ⟨⟨fun x ↦ (isIntegral' F (algebraMap E K x)).tower_bot_of_field⟩, fun x h ↦ ?_⟩
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F K
    x : E
    h : IsSeparable F x
    ⊢ Membership.mem (algebraMap F E).range x
  -/
  rw [IsSeparable, ← minpoly.algebraMap_eq (algebraMap E K).injective] at h
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F K
    x : E
    h : (minpoly F ((algebraMap E K) x)).Separable
    ⊢ Membership.mem (algebraMap F E).range x
  -/
  obtain ⟨y, h⟩ := inseparable F _ h
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F K
    x : E
    h✝ : (minpoly F ((algebraMap E K) x)).Separable
    y : F
    h : Eq ((algebraMap F K) y) ((algebraMap E K) x)
    ⊢ Membership.mem (algebraMap F E).range x
  -/
  exact ⟨y, (algebraMap E K).injective (h.symm ▸ (IsScalarTower.algebraMap_apply F E K y).symm)⟩
  /-
    🎉 no goals
  -/


/-- If `K / E / F` is a field extension tower such that `K / F` is purely inseparable,
then `K / E` is also purely inseparable. -/
theorem IsPurelyInseparable.tower_top [Algebra E K] [IsScalarTower F E K]
    [h : IsPurelyInseparable F K] : IsPurelyInseparable E K := by
  /-
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    h : IsPurelyInseparable F K
    ⊢ IsPurelyInseparable E K
  -/
  obtain ⟨q, _⟩ := ExpChar.exists F
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    h : IsPurelyInseparable F K
    q : Nat
    h✝ : ExpChar F q
    ⊢ IsPurelyInseparable E K
  -/
  haveI := expChar_of_injective_algebraMap (algebraMap F E).injective q
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    h : IsPurelyInseparable F K
    q : Nat
    h✝ : ExpChar F q
    this : ExpChar E q
    ⊢ IsPurelyInseparable E K
  -/
  rw [isPurelyInseparable_iff_pow_mem _ q] at h ⊢
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    q : Nat
    h : ∀ (x : K), Exists fun n => Membership.mem (algebraMap F K).range (HPow.hPo …
    h✝ : ExpChar F q
    this : ExpChar E q
    ⊢ ∀ (x : K), Exists fun n => Membership.mem (algebraMap E K).range (HPow.hPow  …
  -/
  intro x
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    q : Nat
    h : ∀ (x : K), Exists fun n => Membership.mem (algebraMap F K).range (HPow.hPo …
    h✝ : ExpChar F q
    this : ExpChar E q
    x : K
    ⊢ Exists fun n => Membership.mem (algebraMap E K).range (HPow.hPow x (HPow.hPo …
  -/
  obtain ⟨n, y, h⟩ := h x
  /-
    case intro.intro.intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    q : Nat
    h✝¹ : ∀ (x : K), Exists fun n => Membership.mem (algebraMap F K).range (HPow.h …
    h✝ : ExpChar F q
    this : ExpChar E q
    x : K
    n : Nat
    y : F
    h : Eq ((algebraMap F K) y) (HPow.hPow x (HPow.hPow q n))
    ⊢ Exists fun n => Membership.mem (algebraMap E K).range (HPow.hPow x (HPow.hPo …
  -/
  exact ⟨n, (algebraMap F E) y, h.symm ▸ (IsScalarTower.algebraMap_apply F E K y).symm⟩
  /-
    🎉 no goals
  -/


/-- If `E / F` and `K / E` are both purely inseparable extensions, then `K / F` is also
purely inseparable. -/
@[stacks 02JJ "See also 00GM"]
theorem IsPurelyInseparable.trans [Algebra E K] [IsScalarTower F E K]
    [h1 : IsPurelyInseparable F E] [h2 : IsPurelyInseparable E K] : IsPurelyInseparable F K := by
  /-
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    h1 : IsPurelyInseparable F E
    h2 : IsPurelyInseparable E K
    ⊢ IsPurelyInseparable F K
  -/
  obtain ⟨q, _⟩ := ExpChar.exists F
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    h1 : IsPurelyInseparable F E
    h2 : IsPurelyInseparable E K
    q : Nat
    h✝ : ExpChar F q
    ⊢ IsPurelyInseparable F K
  -/
  haveI := expChar_of_injective_algebraMap (algebraMap F E).injective q
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    h1 : IsPurelyInseparable F E
    h2 : IsPurelyInseparable E K
    q : Nat
    h✝ : ExpChar F q
    this : ExpChar E q
    ⊢ IsPurelyInseparable F K
  -/
  rw [isPurelyInseparable_iff_pow_mem _ q] at h1 h2 ⊢
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    q : Nat
    h2 : ∀ (x : K), Exists fun n => Membership.mem (algebraMap E K).range (HPow.hP …
    h1 : ∀ (x : E), Exists fun n => Membership.mem (algebraMap F E).range (HPow.hP …
    h✝ : ExpChar F q
    this : ExpChar E q
    ⊢ ∀ (x : K), Exists fun n => Membership.mem (algebraMap F K).range (HPow.hPow  …
  -/
  intro x
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    q : Nat
    h2 : ∀ (x : K), Exists fun n => Membership.mem (algebraMap E K).range (HPow.hP …
    h1 : ∀ (x : E), Exists fun n => Membership.mem (algebraMap F E).range (HPow.hP …
    h✝ : ExpChar F q
    this : ExpChar E q
    x : K
    ⊢ Exists fun n => Membership.mem (algebraMap F K).range (HPow.hPow x (HPow.hPo …
  -/
  obtain ⟨n, y, h2⟩ := h2 x
  /-
    case intro.intro.intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    q : Nat
    h2✝ : ∀ (x : K), Exists fun n => Membership.mem (algebraMap E K).range (HPow.h …
    h1 : ∀ (x : E), Exists fun n => Membership.mem (algebraMap F E).range (HPow.hP …
    h✝ : ExpChar F q
    this : ExpChar E q
    x : K
    n : Nat
    y : E
    h2 : Eq ((algebraMap E K) y) (HPow.hPow x (HPow.hPow q n))
    ⊢ Exists fun n => Membership.mem (algebraMap F K).range (HPow.hPow x (HPow.hPo …
  -/
  obtain ⟨m, z, h1⟩ := h1 y
  /-
    case intro.intro.intro.intro.intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    q : Nat
    h2✝ : ∀ (x : K), Exists fun n => Membership.mem (algebraMap E K).range (HPow.h …
    h1✝ : ∀ (x : E), Exists fun n => Membership.mem (algebraMap F E).range (HPow.h …
    h✝ : ExpChar F q
    this : ExpChar E q
    x : K
    n : Nat
    y : E
    h2 : Eq ((algebraMap E K) y) (HPow.hPow x (HPow.hPow q n))
    m : Nat
    z : F
    h1 : Eq ((algebraMap F E) z) (HPow.hPow y (HPow.hPow q m))
    ⊢ Exists fun n => Membership.mem (algebraMap F K).range (HPow.hPow x (HPow.hPo …
  -/
  refine ⟨n + m, z, ?_⟩
  /-
    case intro.intro.intro.intro.intro
    F : Type u
    E : Type v
    inst✝⁶ : Field F
    inst✝⁵ : Field E
    inst✝⁴ : Algebra F E
    K : Type w
    inst✝³ : Field K
    inst✝² : Algebra F K
    inst✝¹ : Algebra E K
    inst✝ : IsScalarTower F E K
    q : Nat
    h2✝ : ∀ (x : K), Exists fun n => Membership.mem (algebraMap E K).range (HPow.h …
    h1✝ : ∀ (x : E), Exists fun n => Membership.mem (algebraMap F E).range (HPow.h …
    h✝ : ExpChar F q
    this : ExpChar E q
    x : K
    n : Nat
    y : E
    h2 : Eq ((algebraMap E K) y) (HPow.hPow x (HPow.hPow q n))
    m : Nat
    z : F
    h1 : Eq ((algebraMap F E) z) (HPow.hPow y (HPow.hPow q m))
    ⊢ Eq ((algebraMap F K) z) (HPow.hPow x (HPow.hPow q (HAdd.hAdd n m)))
  -/
  rw [IsScalarTower.algebraMap_apply F E K, h1, map_pow, h2, ← pow_mul, ← pow_add]
  /-
    🎉 no goals
  -/


instance isPurelyInseparable_tower_bot [IsPurelyInseparable F K] : IsPurelyInseparable F M :=
  IsPurelyInseparable.tower_bot F M K


instance isPurelyInseparable_tower_top [IsPurelyInseparable F K] : IsPurelyInseparable M K :=
  IsPurelyInseparable.tower_top F M K


/-- A field extension `E / F` is purely inseparable if and only if for every element `x` of `E`,
its minimal polynomial has separable degree one. -/
theorem isPurelyInseparable_iff_natSepDegree_eq_one :
    IsPurelyInseparable F E ↔ ∀ x : E, (minpoly F x).natSepDegree = 1 := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Iff (IsPurelyInseparable F E) (∀ (x : E), Eq (minpoly F x).natSepDegree 1)
  -/
  obtain ⟨q, _⟩ := ExpChar.exists F
  /-
    case intro
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    q : Nat
    h✝ : ExpChar F q
    ⊢ Iff (IsPurelyInseparable F E) (∀ (x : E), Eq (minpoly F x).natSepDegree 1)
  -/
  simp_rw [isPurelyInseparable_iff_pow_mem F q, minpoly.natSepDegree_eq_one_iff_pow_mem q]
  /-
    🎉 no goals
  -/


theorem IsPurelyInseparable.natSepDegree_eq_one [IsPurelyInseparable F E] (x : E) :
    (minpoly F x).natSepDegree = 1 :=
  (isPurelyInseparable_iff_natSepDegree_eq_one F).1 ‹_› x


/-- A field extension `E / F` of exponential characteristic `q` is purely inseparable
if and only if for every element `x` of `E`, the minimal polynomial of `x` over `F` is of form
`X ^ (q ^ n) - y` for some natural number `n` and some element `y` of `F`. -/
theorem isPurelyInseparable_iff_minpoly_eq_X_pow_sub_C (q : ℕ) [hF : ExpChar F q] :
    IsPurelyInseparable F E ↔ ∀ x : E, ∃ (n : ℕ) (y : F), minpoly F x = X ^ q ^ n - C y := by
  simp_rw [isPurelyInseparable_iff_natSepDegree_eq_one,
    minpoly.natSepDegree_eq_one_iff_eq_X_pow_sub_C q]


theorem IsPurelyInseparable.minpoly_eq_X_pow_sub_C (q : ℕ) [ExpChar F q] [IsPurelyInseparable F E]
    (x : E) : ∃ (n : ℕ) (y : F), minpoly F x = X ^ q ^ n - C y :=
  (isPurelyInseparable_iff_minpoly_eq_X_pow_sub_C F q).1 ‹_› x


/-- A field extension `E / F` of exponential characteristic `q` is purely inseparable
if and only if for every element `x` of `E`, the minimal polynomial of `x` over `F` is of form
`(X - x) ^ (q ^ n)` for some natural number `n`. -/
theorem isPurelyInseparable_iff_minpoly_eq_X_sub_C_pow (q : ℕ) [hF : ExpChar F q] :
    IsPurelyInseparable F E ↔
    ∀ x : E, ∃ n : ℕ, (minpoly F x).map (algebraMap F E) = (X - C x) ^ q ^ n := by
  simp_rw [isPurelyInseparable_iff_natSepDegree_eq_one,
    minpoly.natSepDegree_eq_one_iff_eq_X_sub_C_pow q]


theorem IsPurelyInseparable.minpoly_eq_X_sub_C_pow (q : ℕ) [ExpChar F q] [IsPurelyInseparable F E]
    (x : E) : ∃ n : ℕ, (minpoly F x).map (algebraMap F E) = (X - C x) ^ q ^ n :=
  (isPurelyInseparable_iff_minpoly_eq_X_sub_C_pow F q).1 ‹_› x


variable {F E} in
/-- If an extension has finite separable degree one, then it is purely inseparable. -/
theorem isPurelyInseparable_of_finSepDegree_eq_one
    (hdeg : finSepDegree F E = 1) : IsPurelyInseparable F E := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    hdeg : Eq (Field.finSepDegree F E) 1
    ⊢ IsPurelyInseparable F E
  -/
  by_cases H : Algebra.IsAlgebraic F E
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hdeg : Eq (Field.finSepDegree F E) 1
      H : Algebra.IsAlgebraic F E
      ⊢ IsPurelyInseparable F E
    -/
  · rw [isPurelyInseparable_iff]
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hdeg : Eq (Field.finSepDegree F E) 1
      H : Algebra.IsAlgebraic F E
      ⊢ ∀ (x : E), And (IsIntegral F x) (IsSeparable F x → Membership.mem (algebraMa …
    -/
    refine fun x ↦ ⟨Algebra.IsIntegral.isIntegral x, fun hsep ↦ ?_⟩
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hdeg : Eq (Field.finSepDegree F E) 1
      H : Algebra.IsAlgebraic F E
      x : E
      hsep : IsSeparable F x
      ⊢ Membership.mem (algebraMap F E).range x
    -/
    have : Algebra.IsAlgebraic F⟮x⟯ E := Algebra.IsAlgebraic.tower_top (K := F) F⟮x⟯
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hdeg : Eq (Field.finSepDegree F E) 1
      H : Algebra.IsAlgebraic F E
      x : E
      hsep : IsSeparable F x
      this : Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (IntermediateFie …
      ⊢ Membership.mem (algebraMap F E).range x
    -/
    have := finSepDegree_mul_finSepDegree_of_isAlgebraic F F⟮x⟯ E
    rw [hdeg, mul_eq_one, (finSepDegree_adjoin_simple_eq_finrank_iff F E x
        (Algebra.IsAlgebraic.isAlgebraic x)).2 hsep,
      IntermediateField.finrank_eq_one_iff] at this
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hdeg : Eq (Field.finSepDegree F E) 1
      H : Algebra.IsAlgebraic F E
      x : E
      hsep : IsSeparable F x
      this✝ : Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (IntermediateFi …
      this : And (Eq (IntermediateField.adjoin F (Singleton.singleton x)) Bot.bot) ( …
      ⊢ Membership.mem (algebraMap F E).range x
    -/
    simpa only [this.1] using mem_adjoin_simple_self F x
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hdeg : Eq (Field.finSepDegree F E) 1
      H : Not (Algebra.IsAlgebraic F E)
      ⊢ IsPurelyInseparable F E
    -/
  · rw [← Algebra.transcendental_iff_not_isAlgebraic] at H
    /-
      case neg
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hdeg : Eq (Field.finSepDegree F E) 1
      H : Algebra.Transcendental F E
      ⊢ IsPurelyInseparable F E
    -/
    simp [finSepDegree_eq_zero_of_transcendental F E] at hdeg
    /-
      🎉 no goals
    -/


/-- If `E / F` is purely inseparable, then for any reduced ring `L`, the map `(E →+* L) → (F →+* L)`
induced by `algebraMap F E` is injective. In particular, a purely inseparable field extension
is an epimorphism in the category of fields. -/
theorem injective_comp_algebraMap [CommRing L] [IsReduced L] :
    Function.Injective fun f : E →+* L ↦ f.comp (algebraMap F E) := fun f g heq ↦ by
  /-
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    inst✝² : IsPurelyInseparable F E
    L : Type u_2
    inst✝¹ : CommRing L
    inst✝ : IsReduced L
    f g : RingHom E L
    heq : Eq ((fun f => f.comp (algebraMap F E)) f) ((fun f => f.comp (algebraMap  …
    ⊢ Eq f g
  -/
  ext x
  /-
    case a
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    inst✝² : IsPurelyInseparable F E
    L : Type u_2
    inst✝¹ : CommRing L
    inst✝ : IsReduced L
    f g : RingHom E L
    heq : Eq ((fun f => f.comp (algebraMap F E)) f) ((fun f => f.comp (algebraMap  …
    x : E
    ⊢ Eq (f x) (g x)
  -/
  let q := ringExpChar F
  /-
    case a
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    inst✝² : IsPurelyInseparable F E
    L : Type u_2
    inst✝¹ : CommRing L
    inst✝ : IsReduced L
    f g : RingHom E L
    heq : Eq ((fun f => f.comp (algebraMap F E)) f) ((fun f => f.comp (algebraMap  …
    x : E
    q : Nat := ringExpChar F
    ⊢ Eq (f x) (g x)
  -/
  obtain ⟨n, y, h⟩ := IsPurelyInseparable.pow_mem F q x
  /-
    case a.intro.intro
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    inst✝² : IsPurelyInseparable F E
    L : Type u_2
    inst✝¹ : CommRing L
    inst✝ : IsReduced L
    f g : RingHom E L
    heq : Eq ((fun f => f.comp (algebraMap F E)) f) ((fun f => f.comp (algebraMap  …
    x : E
    q : Nat := ringExpChar F
    n : Nat
    y : F
    h : Eq ((algebraMap F E) y) (HPow.hPow x (HPow.hPow q n))
    ⊢ Eq (f x) (g x)
  -/
  replace heq := congr($heq y)
  /-
    case a.intro.intro
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    inst✝² : IsPurelyInseparable F E
    L : Type u_2
    inst✝¹ : CommRing L
    inst✝ : IsReduced L
    f g : RingHom E L
    x : E
    q : Nat := ringExpChar F
    n : Nat
    y : F
    h : Eq ((algebraMap F E) y) (HPow.hPow x (HPow.hPow q n))
    heq : Eq (((fun f => f.comp (algebraMap F E)) f) y) (((fun f => f.comp (algebr …
    ⊢ Eq (f x) (g x)
  -/
  simp_rw [RingHom.comp_apply, h, map_pow] at heq
  /-
    case a.intro.intro
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    inst✝² : IsPurelyInseparable F E
    L : Type u_2
    inst✝¹ : CommRing L
    inst✝ : IsReduced L
    f g : RingHom E L
    x : E
    q : Nat := ringExpChar F
    n : Nat
    y : F
    h : Eq ((algebraMap F E) y) (HPow.hPow x (HPow.hPow q n))
    heq : Eq (HPow.hPow (f x) (HPow.hPow q n)) (HPow.hPow (g x) (HPow.hPow q n))
    ⊢ Eq (f x) (g x)
  -/
  nontriviality L
  /-
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    inst✝² : IsPurelyInseparable F E
    L : Type u_2
    inst✝¹ : CommRing L
    inst✝ : IsReduced L
    f g : RingHom E L
    x : E
    q : Nat := ringExpChar F
    n : Nat
    y : F
    h : Eq ((algebraMap F E) y) (HPow.hPow x (HPow.hPow q n))
    heq : Eq (HPow.hPow (f x) (HPow.hPow q n)) (HPow.hPow (g x) (HPow.hPow q n))
    a✝ : Nontrivial L
    ⊢ Eq (f x) (g x)
  -/
  haveI := expChar_of_injective_ringHom (f.comp (algebraMap F E)).injective q
  /-
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    inst✝² : IsPurelyInseparable F E
    L : Type u_2
    inst✝¹ : CommRing L
    inst✝ : IsReduced L
    f g : RingHom E L
    x : E
    q : Nat := ringExpChar F
    n : Nat
    y : F
    h : Eq ((algebraMap F E) y) (HPow.hPow x (HPow.hPow q n))
    heq : Eq (HPow.hPow (f x) (HPow.hPow q n)) (HPow.hPow (g x) (HPow.hPow q n))
    a✝ : Nontrivial L
    this : ExpChar L q
    ⊢ Eq (f x) (g x)
  -/
  exact iterateFrobenius_inj L q n heq
  /-
    🎉 no goals
  -/


theorem injective_restrictDomain [CommRing L] [IsReduced L] [Algebra R L] [IsScalarTower R F E] :
    Function.Injective (AlgHom.restrictDomain (A := R) F (C := E) (D := L)) := fun _ _ eq ↦
  AlgHom.coe_ringHom_injective <| injective_comp_algebraMap F E L <| congr_arg AlgHom.toRingHom eq


instance [Field L] [PerfectField L] [Algebra F L] : Nonempty (E →ₐ[F] L) :=
  nonempty_algHom_of_splits fun x ↦ ⟨IsPurelyInseparable.isIntegral' _ _,
    have ⟨q, _⟩ := ExpChar.exists F
    PerfectField.splits_of_natSepDegree_eq_one (algebraMap F L)
      ((minpoly.natSepDegree_eq_one_iff_eq_X_pow_sub_C q).mpr <|
        IsPurelyInseparable.minpoly_eq_X_pow_sub_C F q x)⟩


theorem bijective_comp_algebraMap [Field L] [PerfectField L] :
    Function.Bijective fun f : E →+* L ↦ f.comp (algebraMap F E) :=
  ⟨injective_comp_algebraMap F E L, fun g ↦ let _ := g.toAlgebra
    ⟨_, (Classical.arbitrary <| E →ₐ[F] L).comp_algebraMap⟩⟩


theorem bijective_restrictDomain [Field L] [PerfectField L] [Algebra R L] [IsScalarTower R F E] :
    Function.Bijective (AlgHom.restrictDomain (A := R) F (C := E) (D := L)) :=
  ⟨injective_restrictDomain F E R L, fun g ↦ let _ := g.toAlgebra
    let f := Classical.arbitrary (E →ₐ[F] L)
    ⟨f.restrictScalars R, AlgHom.coe_ringHom_injective f.comp_algebraMap⟩⟩


/-- If `E / F` is purely inseparable, then for any reduced `F`-algebra `L`, there exists at most one
`F`-algebra homomorphism from `E` to `L`. -/
instance instSubsingletonAlgHomOfIsPurelyInseparable [IsPurelyInseparable F E] (L : Type w)
    [CommRing L] [IsReduced L] [Algebra F L] : Subsingleton (E →ₐ[F] L) where
  allEq f g := AlgHom.coe_ringHom_injective <|
                                                            /-
                                                              F : Type u
                                                              E : Type v
                                                              inst✝⁸ : Field F
                                                              inst✝⁷ : Field E
                                                              inst✝⁶ : Algebra F E
                                                              K : Type w
                                                              inst✝⁵ : Field K
                                                              inst✝⁴ : Algebra F K
                                                              inst✝³ : IsPurelyInseparable F E
                                                              L : Type w
                                                              inst✝² : CommRing L
                                                              inst✝¹ : IsReduced L
                                                              inst✝ : Algebra F L
                                                              f g : AlgHom F E L
                                                              ⊢ Eq ((fun f => f.comp (algebraMap F E)) ↑f) ((fun f => f.comp (algebraMap F E …
                                                            -/
    IsPurelyInseparable.injective_comp_algebraMap F E L (by simp_rw [AlgHom.comp_algebraMap])
                                                            /-
                                                              🎉 no goals
                                                            -/


instance instUniqueAlgHomOfIsPurelyInseparable [IsPurelyInseparable F E] (L : Type w)
    [CommRing L] [IsReduced L] [Algebra F L] [Algebra E L] [IsScalarTower F E L] :
    Unique (E →ₐ[F] L) := uniqueOfSubsingleton (IsScalarTower.toAlgHom F E L)


/-- If `E / F` is purely inseparable, then `Field.Emb F E` has exactly one element. -/
instance instUniqueEmbOfIsPurelyInseparable [IsPurelyInseparable F E] :
    Unique (Emb F E) := instUniqueAlgHomOfIsPurelyInseparable F E _


/-- A purely inseparable extension has finite separable degree one. -/
theorem IsPurelyInseparable.finSepDegree_eq_one [IsPurelyInseparable F E] :
    finSepDegree F E = 1 := Nat.card_unique


/-- A purely inseparable extension has separable degree one. -/
theorem IsPurelyInseparable.sepDegree_eq_one [IsPurelyInseparable F E] :
    sepDegree F E = 1 := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : IsPurelyInseparable F E
    ⊢ Eq (Field.sepDegree F E) 1
  -/
  rw [sepDegree, separableClosure.eq_bot_of_isPurelyInseparable, IntermediateField.rank_bot]
  /-
    🎉 no goals
  -/


/-- A purely inseparable extension has inseparable degree equal to degree. -/
theorem IsPurelyInseparable.insepDegree_eq [IsPurelyInseparable F E] :
    insepDegree F E = Module.rank F E := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : IsPurelyInseparable F E
    ⊢ Eq (Field.insepDegree F E) (Module.rank F E)
  -/
  rw [insepDegree, separableClosure.eq_bot_of_isPurelyInseparable, rank_bot']
  /-
    🎉 no goals
  -/


/-- A purely inseparable extension has finite inseparable degree equal to degree. -/
theorem IsPurelyInseparable.finInsepDegree_eq [IsPurelyInseparable F E] :
    finInsepDegree F E = finrank F E := congr(Cardinal.toNat $(insepDegree_eq F E))


/-- An extension is purely inseparable if and only if it has finite separable degree one. -/
theorem isPurelyInseparable_iff_finSepDegree_eq_one :
    IsPurelyInseparable F E ↔ finSepDegree F E = 1 :=
  ⟨fun _ ↦ IsPurelyInseparable.finSepDegree_eq_one F E,
    fun h ↦ isPurelyInseparable_of_finSepDegree_eq_one h⟩


variable {F E} in
/-- An algebraic extension is purely inseparable if and only if all of its finite dimensional
subextensions are purely inseparable. -/
theorem isPurelyInseparable_iff_fd_isPurelyInseparable [Algebra.IsAlgebraic F E] :
    IsPurelyInseparable F E ↔
    ∀ L : IntermediateField F E, FiniteDimensional F L → IsPurelyInseparable F L := by
  refine ⟨fun _ _ _ ↦ IsPurelyInseparable.tower_bot F _ E,
    fun h ↦ isPurelyInseparable_iff.2 fun x ↦ ?_⟩
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    h : ∀ (L : IntermediateField F E), FiniteDimensional F (Subtype fun x => Membe …
    x : E
    ⊢ And (IsIntegral F x) (IsSeparable F x → Membership.mem (algebraMap F E).rang …
  -/
  have hx : IsIntegral F x := Algebra.IsIntegral.isIntegral x
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    h : ∀ (L : IntermediateField F E), FiniteDimensional F (Subtype fun x => Membe …
    x : E
    hx : IsIntegral F x
    ⊢ And (IsIntegral F x) (IsSeparable F x → Membership.mem (algebraMap F E).rang …
  -/
  refine ⟨hx, fun _ ↦ ?_⟩
  obtain ⟨y, h⟩ := (h _ (adjoin.finiteDimensional hx)).inseparable' _ <|
    show Separable (minpoly F (AdjoinSimple.gen F x)) by rwa [minpoly_eq]
  /-
    case intro
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    h✝ : ∀ (L : IntermediateField F E), FiniteDimensional F (Subtype fun x => Memb …
    x : E
    hx : IsIntegral F x
    x✝ : IsSeparable F x
    y : F
    h : Eq ((algebraMap F (Subtype fun x_1 => Membership.mem (IntermediateField.ad …
    ⊢ Membership.mem (algebraMap F E).range x
  -/
  exact ⟨y, congr_arg (algebraMap _ E) h⟩
  /-
    🎉 no goals
  -/


/-- A purely inseparable extension is normal. -/
instance IsPurelyInseparable.normal [IsPurelyInseparable F E] : Normal F E where
  toIsAlgebraic := isAlgebraic F E
  splits' x := by
    /-
      F : Type u
      E : Type v
      inst✝⁵ : Field F
      inst✝⁴ : Field E
      inst✝³ : Algebra F E
      K : Type w
      inst✝² : Field K
      inst✝¹ : Algebra F K
      inst✝ : IsPurelyInseparable F E
      x : E
      ⊢ Polynomial.Splits (algebraMap F E) (minpoly F x)
    -/
    obtain ⟨n, h⟩ := IsPurelyInseparable.minpoly_eq_X_sub_C_pow F (ringExpChar F) x
    /-
      case intro
      F : Type u
      E : Type v
      inst✝⁵ : Field F
      inst✝⁴ : Field E
      inst✝³ : Algebra F E
      K : Type w
      inst✝² : Field K
      inst✝¹ : Algebra F K
      inst✝ : IsPurelyInseparable F E
      x : E
      n : Nat
      h : Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (HPow.hPow (HSub.hSub P …
      ⊢ Polynomial.Splits (algebraMap F E) (minpoly F x)
    -/
    rw [← splits_id_iff_splits, h]
    /-
      case intro
      F : Type u
      E : Type v
      inst✝⁵ : Field F
      inst✝⁴ : Field E
      inst✝³ : Algebra F E
      K : Type w
      inst✝² : Field K
      inst✝¹ : Algebra F K
      inst✝ : IsPurelyInseparable F E
      x : E
      n : Nat
      h : Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (HPow.hPow (HSub.hSub P …
      ⊢ Polynomial.Splits (RingHom.id E) (HPow.hPow (HSub.hSub Polynomial.X (Polynom …
    -/
    exact splits_pow _ (splits_X_sub_C _) _
    /-
      🎉 no goals
    -/


/-- If `E / F` is algebraic, then `E` is purely inseparable over the
separable closure of `F` in `E`. -/
@[stacks 030K "$E/E_{sep}$ is purely inseparable."]
instance separableClosure.isPurelyInseparable [Algebra.IsAlgebraic F E] :
    IsPurelyInseparable (separableClosure F E) E := isPurelyInseparable_iff.2 fun x ↦ by
  /-
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    K : Type w
    inst✝² : Field K
    inst✝¹ : Algebra F K
    inst✝ : Algebra.IsAlgebraic F E
    x : E
    ⊢ And (IsIntegral (Subtype fun x => Membership.mem (separableClosure F E) x) x …
  -/
  set L := separableClosure F E
  refine ⟨(IsAlgebraic.tower_top L (Algebra.IsAlgebraic.isAlgebraic (R := F) x)).isIntegral,
    fun h ↦ ?_⟩
  /-
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    K : Type w
    inst✝² : Field K
    inst✝¹ : Algebra F K
    inst✝ : Algebra.IsAlgebraic F E
    x : E
    L : IntermediateField F E := separableClosure F E
    h : IsSeparable (Subtype fun x => Membership.mem L x) x
    ⊢ Membership.mem (algebraMap (Subtype fun x => Membership.mem L x) E).range x
  -/
  haveI := (isSeparable_adjoin_simple_iff_isSeparable L E).2 h
  /-
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    K : Type w
    inst✝² : Field K
    inst✝¹ : Algebra F K
    inst✝ : Algebra.IsAlgebraic F E
    x : E
    L : IntermediateField F E := separableClosure F E
    h : IsSeparable (Subtype fun x => Membership.mem L x) x
    this : Algebra.IsSeparable (Subtype fun x => Membership.mem L x) (Subtype fun  …
    ⊢ Membership.mem (algebraMap (Subtype fun x => Membership.mem L x) E).range x
  -/
  haveI : Algebra.IsSeparable F (restrictScalars F L⟮x⟯) := Algebra.IsSeparable.trans F L L⟮x⟯
  /-
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    K : Type w
    inst✝² : Field K
    inst✝¹ : Algebra F K
    inst✝ : Algebra.IsAlgebraic F E
    x : E
    L : IntermediateField F E := separableClosure F E
    h : IsSeparable (Subtype fun x => Membership.mem L x) x
    this✝ : Algebra.IsSeparable (Subtype fun x => Membership.mem L x) (Subtype fun …
    this : Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateF …
    ⊢ Membership.mem (algebraMap (Subtype fun x => Membership.mem L x) E).range x
  -/
  have hx : x ∈ restrictScalars F L⟮x⟯ := mem_adjoin_simple_self _ x
  /-
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    K : Type w
    inst✝² : Field K
    inst✝¹ : Algebra F K
    inst✝ : Algebra.IsAlgebraic F E
    x : E
    L : IntermediateField F E := separableClosure F E
    h : IsSeparable (Subtype fun x => Membership.mem L x) x
    this✝ : Algebra.IsSeparable (Subtype fun x => Membership.mem L x) (Subtype fun …
    this : Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateF …
    hx : Membership.mem (IntermediateField.restrictScalars F (IntermediateField.ad …
    ⊢ Membership.mem (algebraMap (Subtype fun x => Membership.mem L x) E).range x
  -/
  exact ⟨⟨x, mem_separableClosure_iff.2 <| isSeparable_of_mem_isSeparable F E hx⟩, rfl⟩
  /-
    🎉 no goals
  -/


open Cardinal in
theorem Field.Emb.cardinal_separableClosure [Algebra.IsAlgebraic F E] :
    #(Field.Emb F <| separableClosure F E) = #(Field.Emb F E) := by
  rw [← (embProdEmbOfIsAlgebraic F (separableClosure F E) E).cardinal_eq,
    mk_prod, mk_eq_one (Emb _ E), lift_one, mul_one, lift_id]


/-- An intermediate field of `E / F` contains the separable closure of `F` in `E`
if `E` is purely inseparable over it. -/
theorem separableClosure_le (L : IntermediateField F E)
    [h : IsPurelyInseparable L E] : separableClosure F E ≤ L := fun x hx ↦ by
  obtain ⟨y, rfl⟩ := h.inseparable' _ <|
    IsSeparable.tower_top L (mem_separableClosure_iff.1 hx)
  /-
    case intro
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    L : IntermediateField F E
    h : IsPurelyInseparable (Subtype fun x => Membership.mem L x) E
    y : Subtype fun x => Membership.mem L x
    hx : Membership.mem (separableClosure F E) ((algebraMap (Subtype fun x => Memb …
    ⊢ Membership.mem L ((algebraMap (Subtype fun x => Membership.mem L x) E) y)
  -/
  exact y.2
  /-
    🎉 no goals
  -/


/-- If `E / F` is algebraic, then an intermediate field of `E / F` contains the
separable closure of `F` in `E` if and only if `E` is purely inseparable over it. -/
theorem separableClosure_le_iff [Algebra.IsAlgebraic F E] (L : IntermediateField F E) :
    separableClosure F E ≤ L ↔ IsPurelyInseparable L E := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    L : IntermediateField F E
    ⊢ Iff (LE.le (separableClosure F E) L) (IsPurelyInseparable (Subtype fun x =>  …
  -/
  refine ⟨fun h ↦ ?_, fun _ ↦ separableClosure_le F E L⟩
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    L : IntermediateField F E
    h : LE.le (separableClosure F E) L
    ⊢ IsPurelyInseparable (Subtype fun x => Membership.mem L x) E
  -/
  have := separableClosure.isPurelyInseparable F E
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    L : IntermediateField F E
    h : LE.le (separableClosure F E) L
    this : IsPurelyInseparable (Subtype fun x => Membership.mem (separableClosure  …
    ⊢ IsPurelyInseparable (Subtype fun x => Membership.mem L x) E
  -/
  letI := (inclusion h).toAlgebra
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    L : IntermediateField F E
    h : LE.le (separableClosure F E) L
    this✝ : IsPurelyInseparable (Subtype fun x => Membership.mem (separableClosure …
    this : Algebra (Subtype fun x => Membership.mem (separableClosure F E) x) (Sub …
    ⊢ IsPurelyInseparable (Subtype fun x => Membership.mem L x) E
  -/
  letI : SMul (separableClosure F E) L := Algebra.toSMul
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    L : IntermediateField F E
    h : LE.le (separableClosure F E) L
    this✝¹ : IsPurelyInseparable (Subtype fun x => Membership.mem (separableClosur …
    this✝ : Algebra (Subtype fun x => Membership.mem (separableClosure F E) x) (Su …
    this : SMul (Subtype fun x => Membership.mem (separableClosure F E) x) (Subtyp …
    ⊢ IsPurelyInseparable (Subtype fun x => Membership.mem L x) E
  -/
  haveI : IsScalarTower (separableClosure F E) L E := IsScalarTower.of_algebraMap_eq (congrFun rfl)
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    L : IntermediateField F E
    h : LE.le (separableClosure F E) L
    this✝² : IsPurelyInseparable (Subtype fun x => Membership.mem (separableClosur …
    this✝¹ : Algebra (Subtype fun x => Membership.mem (separableClosure F E) x) (S …
    this✝ : SMul (Subtype fun x => Membership.mem (separableClosure F E) x) (Subty …
    this : IsScalarTower (Subtype fun x => Membership.mem (separableClosure F E) x …
    ⊢ IsPurelyInseparable (Subtype fun x => Membership.mem L x) E
  -/
  exact IsPurelyInseparable.tower_top (separableClosure F E) L E
  /-
    🎉 no goals
  -/


/-- If an intermediate field of `E / F` is separable over `F`, and `E` is purely inseparable
over it, then it is equal to the separable closure of `F` in `E`. -/
theorem eq_separableClosure (L : IntermediateField F E)
    [Algebra.IsSeparable F L] [IsPurelyInseparable L E] : L = separableClosure F E :=
  le_antisymm (le_separableClosure F E L) (separableClosure_le F E L)


open separableClosure in
/-- If `E / F` is algebraic, then an intermediate field of `E / F` is equal to the separable closure
of `F` in `E` if and only if it is separable over `F`, and `E` is purely inseparable
over it. -/
theorem eq_separableClosure_iff [Algebra.IsAlgebraic F E] (L : IntermediateField F E) :
    L = separableClosure F E ↔ Algebra.IsSeparable F L ∧ IsPurelyInseparable L E :=
      /-
        F : Type u
        E : Type v
        inst✝³ : Field F
        inst✝² : Field E
        inst✝¹ : Algebra F E
        inst✝ : Algebra.IsAlgebraic F E
        L : IntermediateField F E
        ⊢ Eq L (separableClosure F E) → And (Algebra.IsSeparable F (Subtype fun x => M …
      -/
  ⟨by rintro rfl; exact ⟨isSeparable F E, isPurelyInseparable F E⟩,
                  /-
                    🎉 no goals
                  -/
   fun ⟨_, _⟩ ↦ eq_separableClosure F E L⟩


/-- If `L` is an algebraically closed field containing `E`, such that the map
`(E →+* L) → (F →+* L)` induced by `algebraMap F E` is injective, then `E / F` is
purely inseparable. As a corollary, epimorphisms in the category of fields must be
purely inseparable extensions. -/
theorem IsPurelyInseparable.of_injective_comp_algebraMap (L : Type w) [Field L] [IsAlgClosed L]
    [Nonempty (E →+* L)] (h : Function.Injective fun f : E →+* L ↦ f.comp (algebraMap F E)) :
    IsPurelyInseparable F E := by
  /-
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    L : Type w
    inst✝² : Field L
    inst✝¹ : IsAlgClosed L
    inst✝ : Nonempty (RingHom E L)
    h : Function.Injective fun f => f.comp (algebraMap F E)
    ⊢ IsPurelyInseparable F E
  -/
  rw [isPurelyInseparable_iff_finSepDegree_eq_one, finSepDegree, Nat.card_eq_one_iff_unique]
  /-
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    L : Type w
    inst✝² : Field L
    inst✝¹ : IsAlgClosed L
    inst✝ : Nonempty (RingHom E L)
    h : Function.Injective fun f => f.comp (algebraMap F E)
    ⊢ And (Subsingleton (Field.Emb F E)) (Nonempty (Field.Emb F E))
  -/
  letI := (Classical.arbitrary (E →+* L)).toAlgebra
  /-
    F : Type u
    E : Type v
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    L : Type w
    inst✝² : Field L
    inst✝¹ : IsAlgClosed L
    inst✝ : Nonempty (RingHom E L)
    h : Function.Injective fun f => f.comp (algebraMap F E)
    this : Algebra E L := (Classical.arbitrary (RingHom E L)).toAlgebra
    ⊢ And (Subsingleton (Field.Emb F E)) (Nonempty (Field.Emb F E))
  -/
  let j : AlgebraicClosure E →ₐ[E] L := IsAlgClosed.lift
  exact ⟨⟨fun f g ↦ DFunLike.ext' <| j.injective.comp_left (congr_arg (⇑) <|
    @h (j.toRingHom.comp f) (j.toRingHom.comp g) (by ext; simp))⟩, inferInstance⟩


instance isPurelyInseparable_bot : IsPurelyInseparable F (⊥ : IntermediateField F E) :=
  (botEquiv F E).symm.isPurelyInseparable


/-- `F⟮x⟯ / F` is a purely inseparable extension if and only if the minimal polynomial of `x`
has separable degree one. -/
theorem isPurelyInseparable_adjoin_simple_iff_natSepDegree_eq_one {x : E} :
    IsPurelyInseparable F F⟮x⟯ ↔ (minpoly F x).natSepDegree = 1 := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    x : E
    ⊢ Iff (IsPurelyInseparable F (Subtype fun x_1 => Membership.mem (IntermediateF …
  -/
  rw [← le_perfectClosure_iff, adjoin_simple_le_iff, mem_perfectClosure_iff_natSepDegree_eq_one]
  /-
    🎉 no goals
  -/


/-- If `F` is of exponential characteristic `q`, then `F⟮x⟯ / F` is a purely inseparable extension
if and only if `x ^ (q ^ n)` is contained in `F` for some `n : ℕ`. -/
theorem isPurelyInseparable_adjoin_simple_iff_pow_mem (q : ℕ) [hF : ExpChar F q] {x : E} :
    IsPurelyInseparable F F⟮x⟯ ↔ ∃ n : ℕ, x ^ q ^ n ∈ (algebraMap F E).range := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    q : Nat
    hF : ExpChar F q
    x : E
    ⊢ Iff (IsPurelyInseparable F (Subtype fun x_1 => Membership.mem (IntermediateF …
  -/
  rw [← le_perfectClosure_iff, adjoin_simple_le_iff, mem_perfectClosure_iff_pow_mem q]
  /-
    🎉 no goals
  -/


/-- If `F` is of exponential characteristic `q`, then `F(S) / F` is a purely inseparable extension
if and only if for any `x ∈ S`, `x ^ (q ^ n)` is contained in `F` for some `n : ℕ`. -/
theorem isPurelyInseparable_adjoin_iff_pow_mem (q : ℕ) [hF : ExpChar F q] {S : Set E} :
    IsPurelyInseparable F (adjoin F S) ↔ ∀ x ∈ S, ∃ n : ℕ, x ^ q ^ n ∈ (algebraMap F E).range := by
  simp_rw [← le_perfectClosure_iff, adjoin_le_iff, ← mem_perfectClosure_iff_pow_mem q,
    Set.subset_def, SetLike.mem_coe]


/-- A compositum of two purely inseparable extensions is purely inseparable. -/
instance isPurelyInseparable_sup (L1 L2 : IntermediateField F E)
    [h1 : IsPurelyInseparable F L1] [h2 : IsPurelyInseparable F L2] :
    IsPurelyInseparable F (L1 ⊔ L2 : IntermediateField F E) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    L1 L2 : IntermediateField F E
    h1 : IsPurelyInseparable F (Subtype fun x => Membership.mem L1 x)
    h2 : IsPurelyInseparable F (Subtype fun x => Membership.mem L2 x)
    ⊢ IsPurelyInseparable F (Subtype fun x => Membership.mem (Max.max L1 L2) x)
  -/
  rw [← le_perfectClosure_iff] at h1 h2 ⊢
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    L1 L2 : IntermediateField F E
    h1 : LE.le L1 (perfectClosure F E)
    h2 : LE.le L2 (perfectClosure F E)
    ⊢ LE.le (Max.max L1 L2) (perfectClosure F E)
  -/
  exact sup_le h1 h2
  /-
    🎉 no goals
  -/


/-- A compositum of purely inseparable extensions is purely inseparable. -/
instance isPurelyInseparable_iSup {ι : Sort*} {t : ι → IntermediateField F E}
    [h : ∀ i, IsPurelyInseparable F (t i)] :
    IsPurelyInseparable F (⨆ i, t i : IntermediateField F E) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ι : Sort u_1
    t : ι → IntermediateField F E
    h : ∀ (i : ι), IsPurelyInseparable F (Subtype fun x => Membership.mem (t i) x)
    ⊢ IsPurelyInseparable F (Subtype fun x => Membership.mem (iSup fun i => t i) x)
  -/
  simp_rw [← le_perfectClosure_iff] at h ⊢
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ι : Sort u_1
    t : ι → IntermediateField F E
    h : ∀ (i : ι), LE.le (t i) (perfectClosure F E)
    ⊢ LE.le (iSup fun i => t i) (perfectClosure F E)
  -/
  exact iSup_le h
  /-
    🎉 no goals
  -/


/-- If `F` is a field of exponential characteristic `q`, `F(S) / F` is separable, then
`F(S) = F(S ^ (q ^ n))` for any natural number `n`. -/
theorem adjoin_eq_adjoin_pow_expChar_pow_of_isSeparable (S : Set E)
    [Algebra.IsSeparable F (adjoin F S)] (q : ℕ) [ExpChar F q] (n : ℕ) :
    adjoin F S = adjoin F ((· ^ q ^ n) '' S) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    S : Set E
    inst✝¹ : Algebra.IsSeparable F (Subtype fun x => Membership.mem (IntermediateF …
    q : Nat
    inst✝ : ExpChar F q
    n : Nat
    ⊢ Eq (IntermediateField.adjoin F S) (IntermediateField.adjoin F (Set.image (fu …
  -/
  set L := adjoin F S
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    S : Set E
    q : Nat
    inst✝¹ : ExpChar F q
    n : Nat
    L : IntermediateField F E := IntermediateField.adjoin F S
    inst✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
    ⊢ Eq L (IntermediateField.adjoin F (Set.image (fun x => HPow.hPow x (HPow.hPow …
  -/
  set M := adjoin F ((· ^ q ^ n) '' S)
  have hi : M ≤ L := by
    rw [adjoin_le_iff]
    rintro _ ⟨y, hy, rfl⟩
    exact pow_mem (subset_adjoin F S hy) _
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    S : Set E
    q : Nat
    inst✝¹ : ExpChar F q
    n : Nat
    L : IntermediateField F E := IntermediateField.adjoin F S
    inst✝ : Algebra.IsSeparable F (Subtype fun x => Membership.mem L x)
    M : IntermediateField F E := IntermediateField.adjoin F (Set.image (fun x => H …
    hi : LE.le M L
    ⊢ Eq L M
  -/
  letI := (inclusion hi).toAlgebra
  haveI : Algebra.IsSeparable M (extendScalars hi) :=
    Algebra.isSeparable_tower_top_of_isSeparable F M L
  haveI : IsPurelyInseparable M (extendScalars hi) := by
    haveI := expChar_of_injective_algebraMap (algebraMap F M).injective q
    rw [extendScalars_adjoin hi, isPurelyInseparable_adjoin_iff_pow_mem M _ q]
    exact fun x hx ↦ ⟨n, ⟨x ^ q ^ n, subset_adjoin F _ ⟨x, hx, rfl⟩⟩, rfl⟩
  simpa only [extendScalars_restrictScalars, restrictScalars_bot_eq_self] using congr_arg
    (restrictScalars F) (extendScalars hi).eq_bot_of_isPurelyInseparable_of_isSeparable


/-- If `E / F` is a separable field extension of exponential characteristic `q`, then
`F(S) = F(S ^ (q ^ n))` for any subset `S` of `E` and any natural number `n`. -/
theorem adjoin_eq_adjoin_pow_expChar_pow_of_isSeparable' [Algebra.IsSeparable F E] (S : Set E)
    (q : ℕ) [ExpChar F q] (n : ℕ) : adjoin F S = adjoin F ((· ^ q ^ n) '' S) :=
  haveI := Algebra.isSeparable_tower_bot_of_isSeparable F (adjoin F S) E
  adjoin_eq_adjoin_pow_expChar_pow_of_isSeparable F E S q n

-- TODO: prove the converse when `F(S) / F` is finite

/-- If `F` is a field of exponential characteristic `q`, `F(S) / F` is separable, then
`F(S) = F(S ^ q)`. -/
theorem adjoin_eq_adjoin_pow_expChar_of_isSeparable (S : Set E) [Algebra.IsSeparable F (adjoin F S)]
    (q : ℕ) [ExpChar F q] : adjoin F S = adjoin F ((· ^ q) '' S) :=
  pow_one q ▸ adjoin_eq_adjoin_pow_expChar_pow_of_isSeparable F E S q 1


/-- If `E / F` is a separable field extension of exponential characteristic `q`, then
`F(S) = F(S ^ q)` for any subset `S` of `E`. -/
theorem adjoin_eq_adjoin_pow_expChar_of_isSeparable' [Algebra.IsSeparable F E] (S : Set E)
    (q : ℕ) [ExpChar F q] : adjoin F S = adjoin F ((· ^ q) '' S) :=
  pow_one q ▸ adjoin_eq_adjoin_pow_expChar_pow_of_isSeparable' F E S q 1


/-- If `E / F` is a separable extension of exponential characteristic `q`, if `{ u_i }` is a family
of elements of `E` which `F`-linearly spans `E`, then `{ u_i ^ (q ^ n) }` also `F`-linearly spans
`E` for any natural number `n`. -/
theorem Field.span_map_pow_expChar_pow_eq_top_of_isSeparable [Algebra.IsSeparable F E]
    (h : Submodule.span F (Set.range v) = ⊤) :
    Submodule.span F (Set.range (v · ^ q ^ n)) = ⊤ := by
  erw [← Algebra.top_toSubmodule, ← top_toSubalgebra, ← adjoin_univ,
    adjoin_eq_adjoin_pow_expChar_pow_of_isSeparable' F E _ q n,
    adjoin_algebraic_toSubalgebra fun x _ ↦ Algebra.IsAlgebraic.isAlgebraic x,
    Set.image_univ, Algebra.adjoin_eq_span, (powMonoidHom _).mrange.closure_eq]
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    inst✝ : Algebra.IsSeparable F E
    h : Eq (Submodule.span F (Set.range v)) Top.top
    ⊢ Eq (Submodule.span F (Set.range fun x => HPow.hPow (v x) (HPow.hPow q n))) ( …
  -/
  refine (Submodule.span_mono <| Set.range_comp_subset_range _ _).antisymm (Submodule.span_le.2 ?_)
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    inst✝ : Algebra.IsSeparable F E
    h : Eq (Submodule.span F (Set.range v)) Top.top
    ⊢ HasSubset.Subset (Set.range (DivisionSemiring.toSemiring.8 (HPow.hPow q n))) …
  -/
  rw [Set.range_comp, ← Set.image_univ]
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    inst✝ : Algebra.IsSeparable F E
    h : Eq (Submodule.span F (Set.range v)) Top.top
    ⊢ HasSubset.Subset (Set.image (DivisionSemiring.toSemiring.8 (HPow.hPow q n))  …
  -/
  haveI := expChar_of_injective_algebraMap (algebraMap F E).injective q
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    inst✝ : Algebra.IsSeparable F E
    h : Eq (Submodule.span F (Set.range v)) Top.top
    this : ExpChar E q
    ⊢ HasSubset.Subset (Set.image (DivisionSemiring.toSemiring.8 (HPow.hPow q n))  …
  -/
  apply h ▸ Submodule.image_span_subset_span (LinearMap.iterateFrobenius F E q n) _
  /-
    🎉 no goals
  -/


/-- If `E / F` is a finite separable extension of exponential characteristic `q`, if `{ u_i }` is a
family of elements of `E` which is `F`-linearly independent, then `{ u_i ^ (q ^ n) }` is also
`F`-linearly independent for any natural number `n`. A special case of
`LinearIndependent.map_pow_expChar_pow_of_isSeparable`
and is an intermediate result used to prove it. -/
private theorem LinearIndependent.map_pow_expChar_pow_of_fd_isSeparable
    [FiniteDimensional F E] [Algebra.IsSeparable F E]
    (h : LinearIndependent F v) : LinearIndependent F (v · ^ q ^ n) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    inst✝¹ : FiniteDimensional F E
    inst✝ : Algebra.IsSeparable F E
    h : LinearIndependent F v
    ⊢ LinearIndependent F fun x => HPow.hPow (v x) (HPow.hPow q n)
  -/
  have h' := h.coe_range
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    inst✝¹ : FiniteDimensional F E
    inst✝ : Algebra.IsSeparable F E
    h : LinearIndependent F v
    h' : LinearIndependent F Subtype.val
    ⊢ LinearIndependent F fun x => HPow.hPow (v x) (HPow.hPow q n)
  -/
  let ι' := h'.extend (Set.range v).subset_univ
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    inst✝¹ : FiniteDimensional F E
    inst✝ : Algebra.IsSeparable F E
    h : LinearIndependent F v
    h' : LinearIndependent F Subtype.val
    ι' : Set E := h'.extend ⋯
    ⊢ LinearIndependent F fun x => HPow.hPow (v x) (HPow.hPow q n)
  -/
  let b : Basis ι' F E := Basis.extend h'
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    inst✝¹ : FiniteDimensional F E
    inst✝ : Algebra.IsSeparable F E
    h : LinearIndependent F v
    h' : LinearIndependent F Subtype.val
    ι' : Set E := h'.extend ⋯
    b : Basis (↑ι') F E := Basis.extend h'
    ⊢ LinearIndependent F fun x => HPow.hPow (v x) (HPow.hPow q n)
  -/
  letI : Fintype ι' := FiniteDimensional.fintypeBasisIndex b
  have H := linearIndependent_of_top_le_span_of_card_eq_finrank
    (span_map_pow_expChar_pow_eq_top_of_isSeparable q n b.span_eq).ge
    (finrank_eq_card_basis b).symm
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    inst✝¹ : FiniteDimensional F E
    inst✝ : Algebra.IsSeparable F E
    h : LinearIndependent F v
    h' : LinearIndependent F Subtype.val
    ι' : Set E := h'.extend ⋯
    b : Basis (↑ι') F E := Basis.extend h'
    this : Fintype ↑ι' := FiniteDimensional.fintypeBasisIndex b
    H : LinearIndependent F fun x => HPow.hPow (b x) (HPow.hPow q n)
    ⊢ LinearIndependent F fun x => HPow.hPow (v x) (HPow.hPow q n)
  -/
  let f (i : ι) : ι' := ⟨v i, h'.subset_extend _ ⟨i, rfl⟩⟩
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    inst✝¹ : FiniteDimensional F E
    inst✝ : Algebra.IsSeparable F E
    h : LinearIndependent F v
    h' : LinearIndependent F Subtype.val
    ι' : Set E := h'.extend ⋯
    b : Basis (↑ι') F E := Basis.extend h'
    this : Fintype ↑ι' := FiniteDimensional.fintypeBasisIndex b
    H : LinearIndependent F fun x => HPow.hPow (b x) (HPow.hPow q n)
    f : ι → ↑ι' := fun i => ⟨v i, ⋯⟩
    ⊢ LinearIndependent F fun x => HPow.hPow (v x) (HPow.hPow q n)
  -/
  convert H.comp f fun _ _ heq ↦ h.injective (by simpa only [f, Subtype.mk.injEq] using heq)
  /-
    case h.e'_4.h
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    inst✝¹ : FiniteDimensional F E
    inst✝ : Algebra.IsSeparable F E
    h : LinearIndependent F v
    h' : LinearIndependent F Subtype.val
    ι' : Set E := h'.extend ⋯
    b : Basis (↑ι') F E := Basis.extend h'
    this : Fintype ↑ι' := FiniteDimensional.fintypeBasisIndex b
    H : LinearIndependent F fun x => HPow.hPow (b x) (HPow.hPow q n)
    f : ι → ↑ι' := fun i => ⟨v i, ⋯⟩
    x✝ : ι
    ⊢ Eq (HPow.hPow (v x✝) (HPow.hPow q n)) (Function.comp (fun x => HPow.hPow (b  …
  -/
  simp_rw [Function.comp_apply, b]
  /-
    case h.e'_4.h
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    inst✝¹ : FiniteDimensional F E
    inst✝ : Algebra.IsSeparable F E
    h : LinearIndependent F v
    h' : LinearIndependent F Subtype.val
    ι' : Set E := h'.extend ⋯
    b : Basis (↑ι') F E := Basis.extend h'
    this : Fintype ↑ι' := FiniteDimensional.fintypeBasisIndex b
    H : LinearIndependent F fun x => HPow.hPow (b x) (HPow.hPow q n)
    f : ι → ↑ι' := fun i => ⟨v i, ⋯⟩
    x✝ : ι
    ⊢ Eq (HPow.hPow (v x✝) (HPow.hPow q n)) (HPow.hPow ((Basis.extend h') (f x✝))  …
  -/
  rw [Basis.extend_apply_self]
  /-
    🎉 no goals
  -/


/-- If `E / F` is a separable extension of exponential characteristic `q`, if `{ u_i }` is a
family of elements of `E` which is `F`-linearly independent, then `{ u_i ^ (q ^ n) }` is also
`F`-linearly independent for any natural number `n`. -/
theorem LinearIndependent.map_pow_expChar_pow_of_isSeparable [Algebra.IsSeparable F E]
    (h : LinearIndependent F v) : LinearIndependent F (v · ^ q ^ n) := by
  classical
  have halg := Algebra.IsSeparable.isAlgebraic F E
  rw [linearIndependent_iff_finset_linearIndependent] at h ⊢
  intro s
  let E' := adjoin F (s.image v : Set E)
  haveI : FiniteDimensional F E' := finiteDimensional_adjoin
    fun x _ ↦ Algebra.IsIntegral.isIntegral x
  haveI : Algebra.IsSeparable F E' := Algebra.isSeparable_tower_bot_of_isSeparable F E' E
  let v' (i : s) : E' := ⟨v i.1, subset_adjoin F _ (Finset.mem_image.2 ⟨i.1, i.2, rfl⟩)⟩
  have h' : LinearIndependent F v' := (h s).of_comp E'.val.toLinearMap
  exact (h'.map_pow_expChar_pow_of_fd_isSeparable q n).map'
    E'.val.toLinearMap (LinearMap.ker_eq_bot_of_injective E'.val.injective)


/-- If `E / F` is a field extension of exponential characteristic `q`, if `{ u_i }` is a
family of separable elements of `E` which is `F`-linearly independent, then `{ u_i ^ (q ^ n) }`
is also `F`-linearly independent for any natural number `n`. -/
theorem LinearIndependent.map_pow_expChar_pow_of_isSeparable'
    (hsep : ∀ i : ι, IsSeparable F (v i))
    (h : LinearIndependent F v) : LinearIndependent F (v · ^ q ^ n) := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    hsep : ∀ (i : ι), IsSeparable F (v i)
    h : LinearIndependent F v
    ⊢ LinearIndependent F fun x => HPow.hPow (v x) (HPow.hPow q n)
  -/
  let E' := adjoin F (Set.range v)
  haveI : Algebra.IsSeparable F E' := (isSeparable_adjoin_iff_isSeparable F _).2 <| by
    rintro _ ⟨y, rfl⟩; exact hsep y
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    hsep : ∀ (i : ι), IsSeparable F (v i)
    h : LinearIndependent F v
    E' : IntermediateField F E := IntermediateField.adjoin F (Set.range v)
    this : Algebra.IsSeparable F (Subtype fun x => Membership.mem E' x)
    ⊢ LinearIndependent F fun x => HPow.hPow (v x) (HPow.hPow q n)
  -/
  let v' (i : ι) : E' := ⟨v i, subset_adjoin F _ ⟨i, rfl⟩⟩
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    q n : Nat
    hF : ExpChar F q
    ι : Type u_1
    v : ι → E
    hsep : ∀ (i : ι), IsSeparable F (v i)
    h : LinearIndependent F v
    E' : IntermediateField F E := IntermediateField.adjoin F (Set.range v)
    this : Algebra.IsSeparable F (Subtype fun x => Membership.mem E' x)
    v' : ι → Subtype fun x => Membership.mem E' x := fun i => ⟨v i, ⋯⟩
    ⊢ LinearIndependent F fun x => HPow.hPow (v x) (HPow.hPow q n)
  -/
  have h' : LinearIndependent F v' := h.of_comp E'.val.toLinearMap
  exact (h'.map_pow_expChar_pow_of_isSeparable q n).map'
    E'.val.toLinearMap (LinearMap.ker_eq_bot_of_injective E'.val.injective)


/-- If `E / F` is a separable extension of exponential characteristic `q`, if `{ u_i }` is an
`F`-basis of `E`, then `{ u_i ^ (q ^ n) }` is also an `F`-basis of `E`
for any natural number `n`. -/
def Basis.mapPowExpCharPowOfIsSeparable [Algebra.IsSeparable F E]
    (b : Basis ι F E) : Basis ι F E :=
  Basis.mk (b.linearIndependent.map_pow_expChar_pow_of_isSeparable q n)
    (span_map_pow_expChar_pow_eq_top_of_isSeparable q n b.span_eq).ge


/-- If `E` is an algebraic closure of `F`, then `F` is separably closed if and only if `E / F` is
purely inseparable. -/
theorem isSepClosed_iff_isPurelyInseparable_algebraicClosure [IsAlgClosure F E] :
    IsSepClosed F ↔ IsPurelyInseparable F E :=
  ⟨fun _ ↦ inferInstance, fun H ↦ by
    /-
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : IsAlgClosure F E
      H : IsPurelyInseparable F E
      ⊢ IsSepClosed F
    -/
    haveI := IsAlgClosure.isAlgClosed F (K := E)
    /-
      F : Type u
      E : Type v
      inst✝³ : Field F
      inst✝² : Field E
      inst✝¹ : Algebra F E
      inst✝ : IsAlgClosure F E
      H : IsPurelyInseparable F E
      this : IsAlgClosed E
      ⊢ IsSepClosed F
    -/
    rwa [← separableClosure.eq_bot_iff, IsSepClosed.separableClosure_eq_bot_iff] at H⟩
    /-
      🎉 no goals
    -/


variable {F E} in
/-- If `E / F` is an algebraic extension, `F` is separably closed,
then `E` is also separably closed. -/
theorem Algebra.IsAlgebraic.isSepClosed [Algebra.IsAlgebraic F E]
    [IsSepClosed F] : IsSepClosed E :=
  have : Algebra.IsAlgebraic F (AlgebraicClosure E) := Algebra.IsAlgebraic.trans (L := E)
  (isSepClosed_iff_isPurelyInseparable_algebraicClosure E _).mpr
    (IsPurelyInseparable.tower_top F E <| AlgebraicClosure E)


theorem perfectField_of_perfectClosure_eq_bot [h : PerfectField E] (eq : perfectClosure F E = ⊥) :
    PerfectField F := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : PerfectField E
    eq : Eq (perfectClosure F E) Bot.bot
    ⊢ PerfectField F
  -/
  let p := ringExpChar F
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : PerfectField E
    eq : Eq (perfectClosure F E) Bot.bot
    p : Nat := ringExpChar F
    ⊢ PerfectField F
  -/
  haveI := expChar_of_injective_algebraMap (algebraMap F E).injective p
  haveI := PerfectRing.ofSurjective F p fun x ↦ by
    obtain ⟨y, h⟩ := surjective_frobenius E p (algebraMap F E x)
    have : y ∈ perfectClosure F E := ⟨1, x, by rw [← h, pow_one, frobenius_def, ringExpChar.eq F p]⟩
    obtain ⟨z, rfl⟩ := eq ▸ this
    exact ⟨z, (algebraMap F E).injective (by erw [RingHom.map_frobenius, h])⟩
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : PerfectField E
    eq : Eq (perfectClosure F E) Bot.bot
    p : Nat := ringExpChar F
    this✝ : ExpChar E p
    this : PerfectRing F p
    ⊢ PerfectField F
  -/
  exact PerfectRing.toPerfectField F p
  /-
    🎉 no goals
  -/


/-- If `E / F` is a separable extension, `E` is perfect, then `F` is also prefect. -/
theorem perfectField_of_isSeparable_of_perfectField_top [Algebra.IsSeparable F E] [PerfectField E] :
    PerfectField F :=
  perfectField_of_perfectClosure_eq_bot F E (perfectClosure.eq_bot_of_isSeparable F E)


/-- If `E` is an algebraic closure of `F`, then `F` is perfect if and only if `E / F` is
separable. -/
theorem perfectField_iff_isSeparable_algebraicClosure [IsAlgClosure F E] :
    PerfectField F ↔ Algebra.IsSeparable F E :=
  ⟨fun _ ↦ IsSepClosure.separable, fun _ ↦ haveI : IsAlgClosed E := IsAlgClosure.isAlgClosed F
    perfectField_of_isSeparable_of_perfectField_top F E⟩


/-- If `E / F` is algebraic, then the `Field.finSepDegree F E` is equal to `Field.sepDegree F E`
as a natural number. This means that the cardinality of `Field.Emb F E` and the degree of
`(separableClosure F E) / F` are both finite or infinite, and when they are finite, they
coincide. -/
@[stacks 09HJ "`sepDegree` is defined as the right hand side of 09HJ"]
theorem finSepDegree_eq [Algebra.IsAlgebraic F E] :
    finSepDegree F E = Cardinal.toNat (sepDegree F E) := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    ⊢ Eq (Field.finSepDegree F E) (Cardinal.toNat (Field.sepDegree F E))
  -/
  have : Algebra.IsAlgebraic (separableClosure F E) E := Algebra.IsAlgebraic.tower_top (K := F) _
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (separableClosure  …
    ⊢ Eq (Field.finSepDegree F E) (Cardinal.toNat (Field.sepDegree F E))
  -/
  have h := finSepDegree_mul_finSepDegree_of_isAlgebraic F (separableClosure F E) E |>.symm
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (separableClosure  …
    h : Eq (Field.finSepDegree F E) (HMul.hMul (Field.finSepDegree F (Subtype fun  …
    ⊢ Eq (Field.finSepDegree F E) (Cardinal.toNat (Field.sepDegree F E))
  -/
  haveI := separableClosure.isSeparable F E
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsAlgebraic F E
    this✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (separableClosure …
    h : Eq (Field.finSepDegree F E) (HMul.hMul (Field.finSepDegree F (Subtype fun  …
    this : Algebra.IsSeparable F (Subtype fun x => Membership.mem (separableClosur …
    ⊢ Eq (Field.finSepDegree F E) (Cardinal.toNat (Field.sepDegree F E))
  -/
  haveI := separableClosure.isPurelyInseparable F E
  rwa [finSepDegree_eq_finrank_of_isSeparable F (separableClosure F E),
    IsPurelyInseparable.finSepDegree_eq_one (separableClosure F E) E, mul_one] at h


/-- The finite separable degree multiply by the finite inseparable degree is equal
to the (finite) field extension degree. -/
theorem finSepDegree_mul_finInsepDegree : finSepDegree F E * finInsepDegree F E = finrank F E := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Eq (HMul.hMul (Field.finSepDegree F E) (Field.finInsepDegree F E)) (Module.f …
  -/
  by_cases halg : Algebra.IsAlgebraic F E
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      halg : Algebra.IsAlgebraic F E
      ⊢ Eq (HMul.hMul (Field.finSepDegree F E) (Field.finInsepDegree F E)) (Module.f …
    -/
  · have := congr_arg Cardinal.toNat (sepDegree_mul_insepDegree F E)
    /-
      case pos
      F : Type u
      E : Type v
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      halg : Algebra.IsAlgebraic F E
      this : Eq (Cardinal.toNat (HMul.hMul (Field.sepDegree F E) (Field.insepDegree  …
      ⊢ Eq (HMul.hMul (Field.finSepDegree F E) (Field.finInsepDegree F E)) (Module.f …
    -/
    rwa [Cardinal.toNat_mul, ← finSepDegree_eq F E] at this
    /-
      🎉 no goals
    -/
  rw [finInsepDegree, finrank_of_infinite_dimensional (K := F) (V := E) fun _ ↦
      halg (Algebra.IsAlgebraic.of_finite F E),
    finrank_of_infinite_dimensional (K := separableClosure F E) (V := E) fun _ ↦
      halg ((separableClosure.isAlgebraic F E).trans),
    mul_zero]


/-- If `K / E / F` is a field extension tower, such that `E / F` is algebraic and `K / E`
is separable, then `E` adjoin `separableClosure F K` is equal to `K`. It is a special case of
`separableClosure.adjoin_eq_of_isAlgebraic`, and is an intermediate result used to prove it. -/
lemma adjoin_eq_of_isAlgebraic_of_isSeparable [Algebra.IsAlgebraic F E]
    [Algebra.IsSeparable E K] : adjoin E (separableClosure F K : Set K) = ⊤ :=
  top_unique fun x _ ↦ by
    /-
      F : Type u
      E : Type v
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      K : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra E K
      inst✝² : IsScalarTower F E K
      inst✝¹ : Algebra.IsAlgebraic F E
      inst✝ : Algebra.IsSeparable E K
      x : K
      x✝ : Membership.mem Top.top x
      ⊢ Membership.mem (IntermediateField.adjoin E ↑(separableClosure F K)) x
    -/
    set S := separableClosure F K
    /-
      F : Type u
      E : Type v
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      K : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra E K
      inst✝² : IsScalarTower F E K
      inst✝¹ : Algebra.IsAlgebraic F E
      inst✝ : Algebra.IsSeparable E K
      x : K
      x✝ : Membership.mem Top.top x
      S : IntermediateField F K := separableClosure F K
      ⊢ Membership.mem (IntermediateField.adjoin E ↑S) x
    -/
    set L := adjoin E (S : Set K)
    /-
      F : Type u
      E : Type v
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      K : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra E K
      inst✝² : IsScalarTower F E K
      inst✝¹ : Algebra.IsAlgebraic F E
      inst✝ : Algebra.IsSeparable E K
      x : K
      x✝ : Membership.mem Top.top x
      S : IntermediateField F K := separableClosure F K
      L : IntermediateField E K := IntermediateField.adjoin E ↑S
      ⊢ Membership.mem L x
    -/
    have := Algebra.isSeparable_tower_top_of_isSeparable E L K
    /-
      F : Type u
      E : Type v
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      K : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra E K
      inst✝² : IsScalarTower F E K
      inst✝¹ : Algebra.IsAlgebraic F E
      inst✝ : Algebra.IsSeparable E K
      x : K
      x✝ : Membership.mem Top.top x
      S : IntermediateField F K := separableClosure F K
      L : IntermediateField E K := IntermediateField.adjoin E ↑S
      this : Algebra.IsSeparable (Subtype fun x => Membership.mem L x) K
      ⊢ Membership.mem L x
    -/
    let i : S →+* L := Subsemiring.inclusion fun x hx ↦ subset_adjoin E (S : Set K) hx
    /-
      F : Type u
      E : Type v
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      K : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra E K
      inst✝² : IsScalarTower F E K
      inst✝¹ : Algebra.IsAlgebraic F E
      inst✝ : Algebra.IsSeparable E K
      x : K
      x✝ : Membership.mem Top.top x
      S : IntermediateField F K := separableClosure F K
      L : IntermediateField E K := IntermediateField.adjoin E ↑S
      this : Algebra.IsSeparable (Subtype fun x => Membership.mem L x) K
      i : RingHom (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership …
      ⊢ Membership.mem L x
    -/
    let _ : Algebra S L := i.toAlgebra
    /-
      F : Type u
      E : Type v
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      K : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra E K
      inst✝² : IsScalarTower F E K
      inst✝¹ : Algebra.IsAlgebraic F E
      inst✝ : Algebra.IsSeparable E K
      x : K
      x✝¹ : Membership.mem Top.top x
      S : IntermediateField F K := separableClosure F K
      L : IntermediateField E K := IntermediateField.adjoin E ↑S
      this : Algebra.IsSeparable (Subtype fun x => Membership.mem L x) K
      i : RingHom (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership …
      x✝ : Algebra (Subtype fun x => Membership.mem S x) (Subtype fun x => Membershi …
      ⊢ Membership.mem L x
    -/
    let _ : SMul S L := Algebra.toSMul
    /-
      F : Type u
      E : Type v
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      K : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra E K
      inst✝² : IsScalarTower F E K
      inst✝¹ : Algebra.IsAlgebraic F E
      inst✝ : Algebra.IsSeparable E K
      x : K
      x✝² : Membership.mem Top.top x
      S : IntermediateField F K := separableClosure F K
      L : IntermediateField E K := IntermediateField.adjoin E ↑S
      this : Algebra.IsSeparable (Subtype fun x => Membership.mem L x) K
      i : RingHom (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership …
      x✝¹ : Algebra (Subtype fun x => Membership.mem S x) (Subtype fun x => Membersh …
      x✝ : SMul (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership.m …
      ⊢ Membership.mem L x
    -/
    have : IsScalarTower S L K := IsScalarTower.of_algebraMap_eq (congrFun rfl)
    /-
      F : Type u
      E : Type v
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      K : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra E K
      inst✝² : IsScalarTower F E K
      inst✝¹ : Algebra.IsAlgebraic F E
      inst✝ : Algebra.IsSeparable E K
      x : K
      x✝² : Membership.mem Top.top x
      S : IntermediateField F K := separableClosure F K
      L : IntermediateField E K := IntermediateField.adjoin E ↑S
      this✝ : Algebra.IsSeparable (Subtype fun x => Membership.mem L x) K
      i : RingHom (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership …
      x✝¹ : Algebra (Subtype fun x => Membership.mem S x) (Subtype fun x => Membersh …
      x✝ : SMul (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership.m …
      this : IsScalarTower (Subtype fun x => Membership.mem S x) (Subtype fun x => M …
      ⊢ Membership.mem L x
    -/
    have : Algebra.IsAlgebraic F K := Algebra.IsAlgebraic.trans (L := E)
    /-
      F : Type u
      E : Type v
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      K : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra E K
      inst✝² : IsScalarTower F E K
      inst✝¹ : Algebra.IsAlgebraic F E
      inst✝ : Algebra.IsSeparable E K
      x : K
      x✝² : Membership.mem Top.top x
      S : IntermediateField F K := separableClosure F K
      L : IntermediateField E K := IntermediateField.adjoin E ↑S
      this✝¹ : Algebra.IsSeparable (Subtype fun x => Membership.mem L x) K
      i : RingHom (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership …
      x✝¹ : Algebra (Subtype fun x => Membership.mem S x) (Subtype fun x => Membersh …
      x✝ : SMul (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership.m …
      this✝ : IsScalarTower (Subtype fun x => Membership.mem S x) (Subtype fun x =>  …
      this : Algebra.IsAlgebraic F K
      ⊢ Membership.mem L x
    -/
    have : IsPurelyInseparable S K := separableClosure.isPurelyInseparable F K
    /-
      F : Type u
      E : Type v
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      K : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra E K
      inst✝² : IsScalarTower F E K
      inst✝¹ : Algebra.IsAlgebraic F E
      inst✝ : Algebra.IsSeparable E K
      x : K
      x✝² : Membership.mem Top.top x
      S : IntermediateField F K := separableClosure F K
      L : IntermediateField E K := IntermediateField.adjoin E ↑S
      this✝² : Algebra.IsSeparable (Subtype fun x => Membership.mem L x) K
      i : RingHom (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership …
      x✝¹ : Algebra (Subtype fun x => Membership.mem S x) (Subtype fun x => Membersh …
      x✝ : SMul (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership.m …
      this✝¹ : IsScalarTower (Subtype fun x => Membership.mem S x) (Subtype fun x => …
      this✝ : Algebra.IsAlgebraic F K
      this : IsPurelyInseparable (Subtype fun x => Membership.mem S x) K
      ⊢ Membership.mem L x
    -/
    have := IsPurelyInseparable.tower_top S L K
    /-
      F : Type u
      E : Type v
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      K : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra E K
      inst✝² : IsScalarTower F E K
      inst✝¹ : Algebra.IsAlgebraic F E
      inst✝ : Algebra.IsSeparable E K
      x : K
      x✝² : Membership.mem Top.top x
      S : IntermediateField F K := separableClosure F K
      L : IntermediateField E K := IntermediateField.adjoin E ↑S
      this✝³ : Algebra.IsSeparable (Subtype fun x => Membership.mem L x) K
      i : RingHom (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership …
      x✝¹ : Algebra (Subtype fun x => Membership.mem S x) (Subtype fun x => Membersh …
      x✝ : SMul (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership.m …
      this✝² : IsScalarTower (Subtype fun x => Membership.mem S x) (Subtype fun x => …
      this✝¹ : Algebra.IsAlgebraic F K
      this✝ : IsPurelyInseparable (Subtype fun x => Membership.mem S x) K
      this : IsPurelyInseparable (Subtype fun x => Membership.mem L x) K
      ⊢ Membership.mem L x
    -/
    obtain ⟨y, rfl⟩ := IsPurelyInseparable.surjective_algebraMap_of_isSeparable L K x
    /-
      case intro
      F : Type u
      E : Type v
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      K : Type w
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra E K
      inst✝² : IsScalarTower F E K
      inst✝¹ : Algebra.IsAlgebraic F E
      inst✝ : Algebra.IsSeparable E K
      S : IntermediateField F K := separableClosure F K
      L : IntermediateField E K := IntermediateField.adjoin E ↑S
      this✝³ : Algebra.IsSeparable (Subtype fun x => Membership.mem L x) K
      i : RingHom (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership …
      x✝² : Algebra (Subtype fun x => Membership.mem S x) (Subtype fun x => Membersh …
      x✝¹ : SMul (Subtype fun x => Membership.mem S x) (Subtype fun x => Membership. …
      this✝² : IsScalarTower (Subtype fun x => Membership.mem S x) (Subtype fun x => …
      this✝¹ : Algebra.IsAlgebraic F K
      this✝ : IsPurelyInseparable (Subtype fun x => Membership.mem S x) K
      this : IsPurelyInseparable (Subtype fun x => Membership.mem L x) K
      y : Subtype fun x => Membership.mem L x
      x✝ : Membership.mem Top.top ((algebraMap (Subtype fun x => Membership.mem L x) …
      ⊢ Membership.mem L ((algebraMap (Subtype fun x => Membership.mem L x) K) y)
    -/
    exact y.2
    /-
      🎉 no goals
    -/


/-- If `K / E / F` is a field extension tower, such that `E / F` is algebraic, then
`E` adjoin `separableClosure F K` is equal to `separableClosure E K`. -/
theorem adjoin_eq_of_isAlgebraic [Algebra.IsAlgebraic F E] :
    adjoin E (separableClosure F K) = separableClosure E K := by
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsAlgebraic F E
    ⊢ Eq (IntermediateField.adjoin E ↑(separableClosure F K)) (separableClosure E K)
  -/
  set S := separableClosure E K
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsAlgebraic F E
    S : IntermediateField E K := separableClosure E K
    ⊢ Eq (IntermediateField.adjoin E ↑(separableClosure F K)) S
  -/
  have h := congr_arg lift (adjoin_eq_of_isAlgebraic_of_isSeparable (F := F) S)
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsAlgebraic F E
    S : IntermediateField E K := separableClosure E K
    h : Eq (IntermediateField.lift (IntermediateField.adjoin E ↑(separableClosure  …
    ⊢ Eq (IntermediateField.adjoin E ↑(separableClosure F K)) S
  -/
  rw [lift_top, lift_adjoin] at h
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsAlgebraic F E
    S : IntermediateField E K := separableClosure E K
    h : Eq (IntermediateField.adjoin E (Set.image Subtype.val ↑(separableClosure F …
    ⊢ Eq (IntermediateField.adjoin E ↑(separableClosure F K)) S
  -/
  haveI : IsScalarTower F S K := IsScalarTower.of_algebraMap_eq (congrFun rfl)
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsAlgebraic F E
    S : IntermediateField E K := separableClosure E K
    h : Eq (IntermediateField.adjoin E (Set.image Subtype.val ↑(separableClosure F …
    this : IsScalarTower F (Subtype fun x => Membership.mem S x) K
    ⊢ Eq (IntermediateField.adjoin E ↑(separableClosure F K)) S
  -/
  rw [← h, ← map_eq_of_separableClosure_eq_bot F (separableClosure_eq_bot E K)]
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsAlgebraic F E
    S : IntermediateField E K := separableClosure E K
    h : Eq (IntermediateField.adjoin E (Set.image Subtype.val ↑(separableClosure F …
    this : IsScalarTower F (Subtype fun x => Membership.mem S x) K
    ⊢ Eq (IntermediateField.adjoin E ↑(IntermediateField.map (IsScalarTower.toAlgH …
  -/
  simp only [S, coe_map, IsScalarTower.coe_toAlgHom', IntermediateField.algebraMap_apply]
  /-
    🎉 no goals
  -/


variable {F K} in
/-- If `K / E / F` is a field extension tower such that `E / F` is purely inseparable,
if `{ u_i }` is a family of separable elements of `K` which is `F`-linearly independent,
then it is also `E`-linearly independent. -/
theorem LinearIndependent.map_of_isPurelyInseparable_of_isSeparable [IsPurelyInseparable F E]
    {ι : Type*} {v : ι → K} (hsep : ∀ i : ι, IsSeparable F (v i))
    (h : LinearIndependent F v) : LinearIndependent E v := by
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F E
    ι : Type u_1
    v : ι → K
    hsep : ∀ (i : ι), IsSeparable F (v i)
    h : LinearIndependent F v
    ⊢ LinearIndependent E v
  -/
  obtain ⟨q, _⟩ := ExpChar.exists F
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F E
    ι : Type u_1
    v : ι → K
    hsep : ∀ (i : ι), IsSeparable F (v i)
    h : LinearIndependent F v
    q : Nat
    h✝ : ExpChar F q
    ⊢ LinearIndependent E v
  -/
  haveI := expChar_of_injective_algebraMap (algebraMap F K).injective q
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F E
    ι : Type u_1
    v : ι → K
    hsep : ∀ (i : ι), IsSeparable F (v i)
    h : LinearIndependent F v
    q : Nat
    h✝ : ExpChar F q
    this : ExpChar K q
    ⊢ LinearIndependent E v
  -/
  refine linearIndependent_iff.mpr fun l hl ↦ Finsupp.ext fun i ↦ ?_
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F E
    ι : Type u_1
    v : ι → K
    hsep : ∀ (i : ι), IsSeparable F (v i)
    h : LinearIndependent F v
    q : Nat
    h✝ : ExpChar F q
    this : ExpChar K q
    l : Finsupp ι E
    hl : Eq ((Finsupp.linearCombination E v) l) 0
    i : ι
    ⊢ Eq (l i) (0 i)
  -/
  choose f hf using fun i ↦ (isPurelyInseparable_iff_pow_mem F q).1 ‹_› (l i)
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F E
    ι : Type u_1
    v : ι → K
    hsep : ∀ (i : ι), IsSeparable F (v i)
    h : LinearIndependent F v
    q : Nat
    h✝ : ExpChar F q
    this : ExpChar K q
    l : Finsupp ι E
    hl : Eq ((Finsupp.linearCombination E v) l) 0
    i : ι
    f : ι → Nat
    hf : ∀ (i : ι), Membership.mem (algebraMap F E).range (HPow.hPow (l i) (HPow.h …
    ⊢ Eq (l i) (0 i)
  -/
  let n := l.support.sup f
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F E
    ι : Type u_1
    v : ι → K
    hsep : ∀ (i : ι), IsSeparable F (v i)
    h : LinearIndependent F v
    q : Nat
    h✝ : ExpChar F q
    this : ExpChar K q
    l : Finsupp ι E
    hl : Eq ((Finsupp.linearCombination E v) l) 0
    i : ι
    f : ι → Nat
    hf : ∀ (i : ι), Membership.mem (algebraMap F E).range (HPow.hPow (l i) (HPow.h …
    n : Nat := l.support.sup f
    ⊢ Eq (l i) (0 i)
  -/
  have := (expChar_pow_pos F q n).ne'
  replace hf (i : ι) : l i ^ q ^ n ∈ (algebraMap F E).range := by
    by_cases hs : i ∈ l.support
    · convert pow_mem (hf i) (q ^ (n - f i)) using 1
      rw [← pow_mul, ← pow_add, Nat.add_sub_of_le (Finset.le_sup hs)]
    exact ⟨0, by rw [map_zero, Finsupp.not_mem_support_iff.1 hs, zero_pow this]⟩
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F E
    ι : Type u_1
    v : ι → K
    hsep : ∀ (i : ι), IsSeparable F (v i)
    h : LinearIndependent F v
    q : Nat
    h✝ : ExpChar F q
    this✝ : ExpChar K q
    l : Finsupp ι E
    hl : Eq ((Finsupp.linearCombination E v) l) 0
    i : ι
    f : ι → Nat
    n : Nat := l.support.sup f
    this : Ne (HPow.hPow q n) 0
    hf : ∀ (i : ι), Membership.mem (algebraMap F E).range (HPow.hPow (l i) (HPow.h …
    ⊢ Eq (l i) (0 i)
  -/
  choose lF hlF using hf
  let lF₀ := Finsupp.onFinset l.support lF fun i ↦ by
    contrapose!
    refine fun hs ↦ (injective_iff_map_eq_zero _).mp (algebraMap F E).injective _ ?_
    rw [hlF, Finsupp.not_mem_support_iff.1 hs, zero_pow this]
  replace h := linearIndependent_iff.1 (h.map_pow_expChar_pow_of_isSeparable' q n hsep) lF₀ <| by
    replace hl := congr($hl ^ q ^ n)
    rw [linearCombination_apply, Finsupp.sum, sum_pow_char_pow, zero_pow this] at hl
    rw [← hl, linearCombination_apply, onFinset_sum _ (fun _ ↦ by exact zero_smul _ _)]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    simp_rw [Algebra.smul_def, mul_pow, IsScalarTower.algebraMap_apply F E K, hlF, map_pow]
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F E
    ι : Type u_1
    v : ι → K
    hsep : ∀ (i : ι), IsSeparable F (v i)
    q : Nat
    h✝ : ExpChar F q
    this✝ : ExpChar K q
    l : Finsupp ι E
    hl : Eq ((Finsupp.linearCombination E v) l) 0
    i : ι
    f : ι → Nat
    n : Nat := l.support.sup f
    this : Ne (HPow.hPow q n) 0
    lF : ι → F
    hlF : ∀ (i : ι), Eq ((algebraMap F E) (lF i)) (HPow.hPow (l i) (HPow.hPow q n))
    lF₀ : Finsupp ι F := Finsupp.onFinset l.support lF ⋯
    h : Eq lF₀ 0
    ⊢ Eq (l i) (0 i)
  -/
  refine pow_eq_zero ((hlF _).symm.trans ?_)
  /-
    case intro
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F E
    ι : Type u_1
    v : ι → K
    hsep : ∀ (i : ι), IsSeparable F (v i)
    q : Nat
    h✝ : ExpChar F q
    this✝ : ExpChar K q
    l : Finsupp ι E
    hl : Eq ((Finsupp.linearCombination E v) l) 0
    i : ι
    f : ι → Nat
    n : Nat := l.support.sup f
    this : Ne (HPow.hPow q n) 0
    lF : ι → F
    hlF : ∀ (i : ι), Eq ((algebraMap F E) (lF i)) (HPow.hPow (l i) (HPow.hPow q n))
    lF₀ : Finsupp ι F := Finsupp.onFinset l.support lF ⋯
    h : Eq lF₀ 0
    ⊢ Eq ((algebraMap F E) (lF i)) 0
  -/
  convert map_zero (algebraMap F E)
  /-
    case h.e'_2.h.e'_6
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F E
    ι : Type u_1
    v : ι → K
    hsep : ∀ (i : ι), IsSeparable F (v i)
    q : Nat
    h✝ : ExpChar F q
    this✝ : ExpChar K q
    l : Finsupp ι E
    hl : Eq ((Finsupp.linearCombination E v) l) 0
    i : ι
    f : ι → Nat
    n : Nat := l.support.sup f
    this : Ne (HPow.hPow q n) 0
    lF : ι → F
    hlF : ∀ (i : ι), Eq ((algebraMap F E) (lF i)) (HPow.hPow (l i) (HPow.hPow q n))
    lF₀ : Finsupp ι F := Finsupp.onFinset l.support lF ⋯
    h : Eq lF₀ 0
    ⊢ Eq (lF i) 0
  -/
  exact congr($h i)
  /-
    🎉 no goals
  -/


/-- If `K / E / F` is a field extension tower, such that `E / F` is purely inseparable and `K / E`
is separable, then the separable degree of `K / F` is equal to the degree of `K / E`.
It is a special case of `Field.lift_sepDegree_mul_lift_sepDegree_of_isAlgebraic`, and is an
intermediate result used to prove it. -/
lemma sepDegree_eq_of_isPurelyInseparable_of_isSeparable
    [IsPurelyInseparable F E] [Algebra.IsSeparable E K] : sepDegree F K = Module.rank E K := by
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    inst✝¹ : IsPurelyInseparable F E
    inst✝ : Algebra.IsSeparable E K
    ⊢ Eq (Field.sepDegree F K) (Module.rank E K)
  -/
  let S := separableClosure F K
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    inst✝¹ : IsPurelyInseparable F E
    inst✝ : Algebra.IsSeparable E K
    S : IntermediateField F K := separableClosure F K
    ⊢ Eq (Field.sepDegree F K) (Module.rank E K)
  -/
  have h := S.adjoin_rank_le_of_isAlgebraic_right E
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    inst✝¹ : IsPurelyInseparable F E
    inst✝ : Algebra.IsSeparable E K
    S : IntermediateField F K := separableClosure F K
    h : LE.le (Module.rank E (Subtype fun x => Membership.mem (IntermediateField.a …
    ⊢ Eq (Field.sepDegree F K) (Module.rank E K)
  -/
  rw [separableClosure.adjoin_eq_of_isAlgebraic_of_isSeparable K, rank_top'] at h
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    inst✝¹ : IsPurelyInseparable F E
    inst✝ : Algebra.IsSeparable E K
    S : IntermediateField F K := separableClosure F K
    h : LE.le (Module.rank E K) (Module.rank F (Subtype fun x => Membership.mem S  …
    ⊢ Eq (Field.sepDegree F K) (Module.rank E K)
  -/
  obtain ⟨ι, ⟨b⟩⟩ := Basis.exists_basis F S
  exact h.antisymm' (b.mk_eq_rank'' ▸ (b.linearIndependent.map' S.val.toLinearMap
    (LinearMap.ker_eq_bot_of_injective S.val.injective)
    |>.map_of_isPurelyInseparable_of_isSeparable E (fun i ↦
      by simpa only [IsSeparable, minpoly_eq] using Algebra.IsSeparable.isSeparable F (b i))
    |>.cardinal_le_rank))


/-- If `K / E / F` is a field extension tower, such that `E / F` is separable,
then $[E:F] [K:E]_s = [K:F]_s$.
It is a special case of `Field.lift_sepDegree_mul_lift_sepDegree_of_isAlgebraic`, and is an
intermediate result used to prove it. -/
lemma lift_rank_mul_lift_sepDegree_of_isSeparable [Algebra.IsSeparable F E] :
    Cardinal.lift.{w} (Module.rank F E) * Cardinal.lift.{v} (sepDegree E K) =
    Cardinal.lift.{v} (sepDegree F K) := by
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    ⊢ Eq (HMul.hMul (Cardinal.lift.{w, v} (Module.rank F E)) (Cardinal.lift.{v, w} …
  -/
  rw [sepDegree, sepDegree, separableClosure.eq_restrictScalars_of_isSeparable F E K]
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    ⊢ Eq (HMul.hMul (Cardinal.lift.{w, v} (Module.rank F E)) (Cardinal.lift.{v, w} …
  -/
  exact lift_rank_mul_lift_rank F E (separableClosure E K)
  /-
    🎉 no goals
  -/


/-- The same-universe version of `Field.lift_rank_mul_lift_sepDegree_of_isSeparable`. -/
lemma rank_mul_sepDegree_of_isSeparable (K : Type v) [Field K] [Algebra F K]
    [Algebra E K] [IsScalarTower F E K] [Algebra.IsSeparable F E] :
    Module.rank F E * sepDegree E K = sepDegree F K := by
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type v
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsSeparable F E
    ⊢ Eq (HMul.hMul (Module.rank F E) (Field.sepDegree E K)) (Field.sepDegree F K)
  -/
  simpa only [Cardinal.lift_id] using lift_rank_mul_lift_sepDegree_of_isSeparable F E K
  /-
    🎉 no goals
  -/


/-- If `K / E / F` is a field extension tower, such that `E / F` is purely inseparable,
then $[K:F]_s = [K:E]_s$.
It is a special case of `Field.lift_sepDegree_mul_lift_sepDegree_of_isAlgebraic`, and is an
intermediate result used to prove it. -/
lemma sepDegree_eq_of_isPurelyInseparable [IsPurelyInseparable F E] :
    sepDegree F K = sepDegree E K := by
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F E
    ⊢ Eq (Field.sepDegree F K) (Field.sepDegree E K)
  -/
  convert sepDegree_eq_of_isPurelyInseparable_of_isSeparable F E (separableClosure E K)
  /-
    case h.e'_2
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : IsPurelyInseparable F E
    ⊢ Eq (Field.sepDegree F K) (Field.sepDegree F (Subtype fun x => Membership.mem …
  -/
  haveI : IsScalarTower F (separableClosure E K) K := IsScalarTower.of_algebraMap_eq (congrFun rfl)
  rw [sepDegree, ← separableClosure.map_eq_of_separableClosure_eq_bot F
    (separableClosure.separableClosure_eq_bot E K)]
  exact (separableClosure F (separableClosure E K)).equivMap
    (IsScalarTower.toAlgHom F (separableClosure E K) K) |>.symm.toLinearEquiv.rank_eq


/-- If `K / E / F` is a field extension tower, such that `E / F` is algebraic, then their
separable degrees satisfy the tower law: $[E:F]_s [K:E]_s = [K:F]_s$. -/
theorem lift_sepDegree_mul_lift_sepDegree_of_isAlgebraic [Algebra.IsAlgebraic F E] :
    Cardinal.lift.{w} (sepDegree F E) * Cardinal.lift.{v} (sepDegree E K) =
    Cardinal.lift.{v} (sepDegree F K) := by
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsAlgebraic F E
    ⊢ Eq (HMul.hMul (Cardinal.lift.{w, v} (Field.sepDegree F E)) (Cardinal.lift.{v …
  -/
  have h := lift_rank_mul_lift_sepDegree_of_isSeparable F (separableClosure F E) K
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsAlgebraic F E
    h : Eq (HMul.hMul (Cardinal.lift.{w, v} (Module.rank F (Subtype fun x => Membe …
    ⊢ Eq (HMul.hMul (Cardinal.lift.{w, v} (Field.sepDegree F E)) (Cardinal.lift.{v …
  -/
  haveI := separableClosure.isPurelyInseparable F E
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsAlgebraic F E
    h : Eq (HMul.hMul (Cardinal.lift.{w, v} (Module.rank F (Subtype fun x => Membe …
    this : IsPurelyInseparable (Subtype fun x => Membership.mem (separableClosure  …
    ⊢ Eq (HMul.hMul (Cardinal.lift.{w, v} (Field.sepDegree F E)) (Cardinal.lift.{v …
  -/
  rwa [sepDegree_eq_of_isPurelyInseparable (separableClosure F E) E K] at h
  /-
    🎉 no goals
  -/


/-- The same-universe version of `Field.lift_sepDegree_mul_lift_sepDegree_of_isAlgebraic`. -/
@[stacks 09HK "Part 1"]
theorem sepDegree_mul_sepDegree_of_isAlgebraic (K : Type v) [Field K] [Algebra F K]
    [Algebra E K] [IsScalarTower F E K] [Algebra.IsAlgebraic F E] :
    sepDegree F E * sepDegree E K = sepDegree F K := by
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type v
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    inst✝ : Algebra.IsAlgebraic F E
    ⊢ Eq (HMul.hMul (Field.sepDegree F E) (Field.sepDegree E K)) (Field.sepDegree  …
  -/
  simpa only [Cardinal.lift_id] using lift_sepDegree_mul_lift_sepDegree_of_isAlgebraic F E K
  /-
    🎉 no goals
  -/


variable {F K} in
/-- If `K / E / F` is a field extension tower, such that `E / F` is purely inseparable, then
for any subset `S` of `K` such that `F(S) / F` is algebraic, the `E(S) / E` and `F(S) / F` have
the same separable degree. -/
theorem IntermediateField.sepDegree_adjoin_eq_of_isAlgebraic_of_isPurelyInseparable
    (S : Set K) [Algebra.IsAlgebraic F (adjoin F S)] [IsPurelyInseparable F E] :
    sepDegree E (adjoin E S) = sepDegree F (adjoin F S) := by
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : Set K
    inst✝¹ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem (IntermediateF …
    inst✝ : IsPurelyInseparable F E
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem (IntermediateField.ad …
  -/
  set M := adjoin F S
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : Set K
    inst✝¹ : IsPurelyInseparable F E
    M : IntermediateField F K := IntermediateField.adjoin F S
    inst✝ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem M x)
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem (IntermediateField.ad …
  -/
  set L := adjoin E S
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : Set K
    inst✝¹ : IsPurelyInseparable F E
    M : IntermediateField F K := IntermediateField.adjoin F S
    inst✝ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem M x)
    L : IntermediateField E K := IntermediateField.adjoin E S
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem L x)) (Field.sepDegre …
  -/
  let E' := (IsScalarTower.toAlgHom F E K).fieldRange
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : Set K
    inst✝¹ : IsPurelyInseparable F E
    M : IntermediateField F K := IntermediateField.adjoin F S
    inst✝ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem M x)
    L : IntermediateField E K := IntermediateField.adjoin E S
    E' : IntermediateField F K := (IsScalarTower.toAlgHom F E K).fieldRange
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem L x)) (Field.sepDegre …
  -/
  let j : E ≃ₐ[F] E' := AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F E K)
  have hi : M ≤ L.restrictScalars F := by
    rw [restrictScalars_adjoin_of_algEquiv (E := K) j rfl, restrictScalars_adjoin]
    exact adjoin.mono _ _ _ Set.subset_union_right
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : Set K
    inst✝¹ : IsPurelyInseparable F E
    M : IntermediateField F K := IntermediateField.adjoin F S
    inst✝ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem M x)
    L : IntermediateField E K := IntermediateField.adjoin E S
    E' : IntermediateField F K := (IsScalarTower.toAlgHom F E K).fieldRange
    j : AlgEquiv F E (Subtype fun x => Membership.mem E' x) := AlgEquiv.ofInjectiv …
    hi : LE.le M (IntermediateField.restrictScalars F L)
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem L x)) (Field.sepDegre …
  -/
  let i : M →+* L := Subsemiring.inclusion hi
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : Set K
    inst✝¹ : IsPurelyInseparable F E
    M : IntermediateField F K := IntermediateField.adjoin F S
    inst✝ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem M x)
    L : IntermediateField E K := IntermediateField.adjoin E S
    E' : IntermediateField F K := (IsScalarTower.toAlgHom F E K).fieldRange
    j : AlgEquiv F E (Subtype fun x => Membership.mem E' x) := AlgEquiv.ofInjectiv …
    hi : LE.le M (IntermediateField.restrictScalars F L)
    i : RingHom (Subtype fun x => Membership.mem M x) (Subtype fun x => Membership …
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem L x)) (Field.sepDegre …
  -/
  letI : Algebra M L := i.toAlgebra
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : Set K
    inst✝¹ : IsPurelyInseparable F E
    M : IntermediateField F K := IntermediateField.adjoin F S
    inst✝ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem M x)
    L : IntermediateField E K := IntermediateField.adjoin E S
    E' : IntermediateField F K := (IsScalarTower.toAlgHom F E K).fieldRange
    j : AlgEquiv F E (Subtype fun x => Membership.mem E' x) := AlgEquiv.ofInjectiv …
    hi : LE.le M (IntermediateField.restrictScalars F L)
    i : RingHom (Subtype fun x => Membership.mem M x) (Subtype fun x => Membership …
    this : Algebra (Subtype fun x => Membership.mem M x) (Subtype fun x => Members …
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem L x)) (Field.sepDegre …
  -/
  letI : SMul M L := Algebra.toSMul
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : Set K
    inst✝¹ : IsPurelyInseparable F E
    M : IntermediateField F K := IntermediateField.adjoin F S
    inst✝ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem M x)
    L : IntermediateField E K := IntermediateField.adjoin E S
    E' : IntermediateField F K := (IsScalarTower.toAlgHom F E K).fieldRange
    j : AlgEquiv F E (Subtype fun x => Membership.mem E' x) := AlgEquiv.ofInjectiv …
    hi : LE.le M (IntermediateField.restrictScalars F L)
    i : RingHom (Subtype fun x => Membership.mem M x) (Subtype fun x => Membership …
    this✝ : Algebra (Subtype fun x => Membership.mem M x) (Subtype fun x => Member …
    this : SMul (Subtype fun x => Membership.mem M x) (Subtype fun x => Membership …
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem L x)) (Field.sepDegre …
  -/
  haveI : IsScalarTower F M L := IsScalarTower.of_algebraMap_eq (congrFun rfl)
  haveI : IsPurelyInseparable M L := by
    change IsPurelyInseparable M (extendScalars hi)
    obtain ⟨q, _⟩ := ExpChar.exists F
    have : extendScalars hi = adjoin M (E' : Set K) := restrictScalars_injective F <| by
      conv_lhs => rw [extendScalars_restrictScalars, restrictScalars_adjoin_of_algEquiv
        (E := K) j rfl, ← adjoin_self F E', adjoin_adjoin_comm]
    rw [this, isPurelyInseparable_adjoin_iff_pow_mem _ _ q]
    rintro x ⟨y, hy⟩
    obtain ⟨n, z, hz⟩ := IsPurelyInseparable.pow_mem F q y
    refine ⟨n, algebraMap F M z, ?_⟩
    rw [← IsScalarTower.algebraMap_apply, IsScalarTower.algebraMap_apply F E K, hz, ← hy, map_pow,
      AlgHom.toRingHom_eq_coe, IsScalarTower.coe_toAlgHom]
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : Set K
    inst✝¹ : IsPurelyInseparable F E
    M : IntermediateField F K := IntermediateField.adjoin F S
    inst✝ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem M x)
    L : IntermediateField E K := IntermediateField.adjoin E S
    E' : IntermediateField F K := (IsScalarTower.toAlgHom F E K).fieldRange
    j : AlgEquiv F E (Subtype fun x => Membership.mem E' x) := AlgEquiv.ofInjectiv …
    hi : LE.le M (IntermediateField.restrictScalars F L)
    i : RingHom (Subtype fun x => Membership.mem M x) (Subtype fun x => Membership …
    this✝² : Algebra (Subtype fun x => Membership.mem M x) (Subtype fun x => Membe …
    this✝¹ : SMul (Subtype fun x => Membership.mem M x) (Subtype fun x => Membersh …
    this✝ : IsScalarTower F (Subtype fun x => Membership.mem M x) (Subtype fun x = …
    this : IsPurelyInseparable (Subtype fun x => Membership.mem M x) (Subtype fun  …
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem L x)) (Field.sepDegre …
  -/
  have h := lift_sepDegree_mul_lift_sepDegree_of_isAlgebraic F E L
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : Set K
    inst✝¹ : IsPurelyInseparable F E
    M : IntermediateField F K := IntermediateField.adjoin F S
    inst✝ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem M x)
    L : IntermediateField E K := IntermediateField.adjoin E S
    E' : IntermediateField F K := (IsScalarTower.toAlgHom F E K).fieldRange
    j : AlgEquiv F E (Subtype fun x => Membership.mem E' x) := AlgEquiv.ofInjectiv …
    hi : LE.le M (IntermediateField.restrictScalars F L)
    i : RingHom (Subtype fun x => Membership.mem M x) (Subtype fun x => Membership …
    this✝² : Algebra (Subtype fun x => Membership.mem M x) (Subtype fun x => Membe …
    this✝¹ : SMul (Subtype fun x => Membership.mem M x) (Subtype fun x => Membersh …
    this✝ : IsScalarTower F (Subtype fun x => Membership.mem M x) (Subtype fun x = …
    this : IsPurelyInseparable (Subtype fun x => Membership.mem M x) (Subtype fun  …
    h : Eq (HMul.hMul (Cardinal.lift.{w, v} (Field.sepDegree F E)) (Cardinal.lift. …
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem L x)) (Field.sepDegre …
  -/
  rw [IsPurelyInseparable.sepDegree_eq_one F E, Cardinal.lift_one, one_mul] at h
  rw [Cardinal.lift_injective h, ← sepDegree_mul_sepDegree_of_isAlgebraic F M L,
    IsPurelyInseparable.sepDegree_eq_one M L, mul_one]


variable {F K} in
/-- If `K / E / F` is a field extension tower, such that `E / F` is purely inseparable, then
for any intermediate field `S` of `K / F` such that `S / F` is algebraic, the `E(S) / E` and
`S / F` have the same separable degree. -/
theorem IntermediateField.sepDegree_adjoin_eq_of_isAlgebraic_of_isPurelyInseparable'
    (S : IntermediateField F K) [Algebra.IsAlgebraic F S] [IsPurelyInseparable F E] :
    sepDegree E (adjoin E (S : Set K)) = sepDegree F S := by
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : IntermediateField F K
    inst✝¹ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem S x)
    inst✝ : IsPurelyInseparable F E
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem (IntermediateField.ad …
  -/
  have : Algebra.IsAlgebraic F (adjoin F (S : Set K)) := by rwa [adjoin_self]
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : IntermediateField F K
    inst✝¹ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem S x)
    inst✝ : IsPurelyInseparable F E
    this : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem (IntermediateFie …
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem (IntermediateField.ad …
  -/
  have := sepDegree_adjoin_eq_of_isAlgebraic_of_isPurelyInseparable (F := F) E (S : Set K)
  /-
    F : Type u
    E : Type v
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    K : Type w
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra E K
    inst✝² : IsScalarTower F E K
    S : IntermediateField F K
    inst✝¹ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem S x)
    inst✝ : IsPurelyInseparable F E
    this✝ : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem (IntermediateFi …
    this : Eq (Field.sepDegree E (Subtype fun x => Membership.mem (IntermediateFie …
    ⊢ Eq (Field.sepDegree E (Subtype fun x => Membership.mem (IntermediateField.ad …
  -/
  rwa [adjoin_self] at this
  /-
    🎉 no goals
  -/


variable {F K} in
/-- If `K / E / F` is a field extension tower, such that `E / F` is purely inseparable, then
for any element `x` of `K` separable over `F`, it has the same minimal polynomials over `F` and
over `E`. -/
theorem minpoly.map_eq_of_isSeparable_of_isPurelyInseparable (x : K)
    (hsep : IsSeparable F x) [IsPurelyInseparable F E] :
    (minpoly F x).map (algebraMap F E) = minpoly E x := by
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    x : K
    hsep : IsSeparable F x
    inst✝ : IsPurelyInseparable F E
    ⊢ Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (minpoly E x)
  -/
  have hi := IsSeparable.isIntegral hsep
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    x : K
    hsep : IsSeparable F x
    inst✝ : IsPurelyInseparable F E
    hi : IsIntegral F x
    ⊢ Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (minpoly E x)
  -/
  have hi' : IsIntegral E x := IsIntegral.tower_top hi
  refine eq_of_monic_of_dvd_of_natDegree_le (monic hi') ((monic hi).map (algebraMap F E))
    (dvd_map_of_isScalarTower F E x) (le_of_eq ?_)
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    x : K
    hsep : IsSeparable F x
    inst✝ : IsPurelyInseparable F E
    hi : IsIntegral F x
    hi' : IsIntegral E x
    ⊢ Eq (Polynomial.map (algebraMap F E) (minpoly F x)).natDegree (minpoly E x).n …
  -/
  have hsep' := IsSeparable.tower_top E hsep
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    x : K
    hsep : IsSeparable F x
    inst✝ : IsPurelyInseparable F E
    hi : IsIntegral F x
    hi' : IsIntegral E x
    hsep' : IsSeparable E x
    ⊢ Eq (Polynomial.map (algebraMap F E) (minpoly F x)).natDegree (minpoly E x).n …
  -/
  haveI := (isSeparable_adjoin_simple_iff_isSeparable _ _).2 hsep
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    x : K
    hsep : IsSeparable F x
    inst✝ : IsPurelyInseparable F E
    hi : IsIntegral F x
    hi' : IsIntegral E x
    hsep' : IsSeparable E x
    this : Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (IntermediateF …
    ⊢ Eq (Polynomial.map (algebraMap F E) (minpoly F x)).natDegree (minpoly E x).n …
  -/
  haveI := (isSeparable_adjoin_simple_iff_isSeparable _ _).2 hsep'
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    x : K
    hsep : IsSeparable F x
    inst✝ : IsPurelyInseparable F E
    hi : IsIntegral F x
    hi' : IsIntegral E x
    hsep' : IsSeparable E x
    this✝ : Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (Intermediate …
    this : Algebra.IsSeparable E (Subtype fun x_1 => Membership.mem (IntermediateF …
    ⊢ Eq (Polynomial.map (algebraMap F E) (minpoly F x)).natDegree (minpoly E x).n …
  -/
  have := Algebra.IsSeparable.isAlgebraic F F⟮x⟯
  /-
    F : Type u
    E : Type v
    inst✝⁷ : Field F
    inst✝⁶ : Field E
    inst✝⁵ : Algebra F E
    K : Type w
    inst✝⁴ : Field K
    inst✝³ : Algebra F K
    inst✝² : Algebra E K
    inst✝¹ : IsScalarTower F E K
    x : K
    hsep : IsSeparable F x
    inst✝ : IsPurelyInseparable F E
    hi : IsIntegral F x
    hi' : IsIntegral E x
    hsep' : IsSeparable E x
    this✝¹ : Algebra.IsSeparable F (Subtype fun x_1 => Membership.mem (Intermediat …
    this✝ : Algebra.IsSeparable E (Subtype fun x_1 => Membership.mem (Intermediate …
    this : Algebra.IsAlgebraic F (Subtype fun x_1 => Membership.mem (IntermediateF …
    ⊢ Eq (Polynomial.map (algebraMap F E) (minpoly F x)).natDegree (minpoly E x).n …
  -/
  have := Algebra.IsSeparable.isAlgebraic E E⟮x⟯
  rw [Polynomial.natDegree_map, ← adjoin.finrank hi, ← adjoin.finrank hi',
    ← finSepDegree_eq_finrank_of_isSeparable F _, ← finSepDegree_eq_finrank_of_isSeparable E _,
    finSepDegree_eq, finSepDegree_eq,
    sepDegree_adjoin_eq_of_isAlgebraic_of_isPurelyInseparable (F := F) E]


variable {F} in
/-- If `E / F` is a purely inseparable field extension, `f` is a separable irreducible polynomial
over `F`, then it is also irreducible over `E`. -/
theorem Polynomial.Separable.map_irreducible_of_isPurelyInseparable {f : F[X]} (hsep : f.Separable)
    (hirr : Irreducible f) [IsPurelyInseparable F E] : Irreducible (f.map (algebraMap F E)) := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    f : Polynomial F
    hsep : f.Separable
    hirr : Irreducible f
    inst✝ : IsPurelyInseparable F E
    ⊢ Irreducible (Polynomial.map (algebraMap F E) f)
  -/
  let K := AlgebraicClosure E
  obtain ⟨x, hx⟩ := IsAlgClosed.exists_aeval_eq_zero K f
    (natDegree_pos_iff_degree_pos.1 hirr.natDegree_pos).ne'
  have ha : Associated f (minpoly F x) := by
    have := isUnit_C.2 (leadingCoeff_ne_zero.2 hirr.ne_zero).isUnit.inv
    exact ⟨this.unit, by rw [IsUnit.unit_spec, minpoly.eq_of_irreducible hirr hx]⟩
  have ha' : Associated (f.map (algebraMap F E)) ((minpoly F x).map (algebraMap F E)) :=
    ha.map (mapRingHom (algebraMap F E)).toMonoidHom
  /-
    case intro
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    f : Polynomial F
    hsep : f.Separable
    hirr : Irreducible f
    inst✝ : IsPurelyInseparable F E
    K : Type v := AlgebraicClosure E
    x : K
    hx : Eq ((Polynomial.aeval x) f) 0
    ha : Associated f (minpoly F x)
    ha' : Associated (Polynomial.map (algebraMap F E) f) (Polynomial.map (algebraM …
    ⊢ Irreducible (Polynomial.map (algebraMap F E) f)
  -/
  have heq := minpoly.map_eq_of_isSeparable_of_isPurelyInseparable E x (ha.separable hsep)
  /-
    case intro
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    f : Polynomial F
    hsep : f.Separable
    hirr : Irreducible f
    inst✝ : IsPurelyInseparable F E
    K : Type v := AlgebraicClosure E
    x : K
    hx : Eq ((Polynomial.aeval x) f) 0
    ha : Associated f (minpoly F x)
    ha' : Associated (Polynomial.map (algebraMap F E) f) (Polynomial.map (algebraM …
    heq : Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (minpoly E x)
    ⊢ Irreducible (Polynomial.map (algebraMap F E) f)
  -/
  rw [ha'.irreducible_iff, heq]
  /-
    case intro
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    f : Polynomial F
    hsep : f.Separable
    hirr : Irreducible f
    inst✝ : IsPurelyInseparable F E
    K : Type v := AlgebraicClosure E
    x : K
    hx : Eq ((Polynomial.aeval x) f) 0
    ha : Associated f (minpoly F x)
    ha' : Associated (Polynomial.map (algebraMap F E) f) (Polynomial.map (algebraM …
    heq : Eq (Polynomial.map (algebraMap F E) (minpoly F x)) (minpoly E x)
    ⊢ Irreducible (minpoly E x)
  -/
  exact minpoly.irreducible (Algebra.IsIntegral.isIntegral x)
  /-
    🎉 no goals
  -/


