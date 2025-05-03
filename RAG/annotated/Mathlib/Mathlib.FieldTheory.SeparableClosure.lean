/-- The (relative) separable closure of `F` in `E`, or called maximal separable subextension
of `E / F`, is defined to be the intermediate field of `E / F` consisting of all separable
elements. The previous results prove that these elements are closed under field operations. -/
@[stacks 09HC]
def separableClosure : IntermediateField F E where
  carrier := {x | IsSeparable F x}
  mul_mem' := isSeparable_mul
  add_mem' := isSeparable_add
  algebraMap_mem' := isSeparable_algebraMap E
  inv_mem' _ := isSeparable_inv


/-- An element is contained in the separable closure of `F` in `E` if and only if
it is a separable element. -/
theorem mem_separableClosure_iff {x : E} :
    x ∈ separableClosure F E ↔ IsSeparable F x := Iff.rfl


/-- If `i` is an `F`-algebra homomorphism from `E` to `K`, then `i x` is contained in
`separableClosure F K` if and only if `x` is contained in `separableClosure F E`. -/
theorem map_mem_separableClosure_iff (i : E →ₐ[F] K) {x : E} :
    i x ∈ separableClosure F K ↔ x ∈ separableClosure F E := by
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
    ⊢ Iff (Membership.mem (separableClosure F K) (i x)) (Membership.mem (separable …
  -/
  simp_rw [mem_separableClosure_iff, IsSeparable, minpoly.algHom_eq i i.injective]
  /-
    🎉 no goals
  -/


/-- If `i` is an `F`-algebra homomorphism from `E` to `K`, then the preimage of
`separableClosure F K` under the map `i` is equal to `separableClosure F E`. -/
theorem separableClosure.comap_eq_of_algHom (i : E →ₐ[F] K) :
    (separableClosure F K).comap i = separableClosure F E := by
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
    ⊢ Eq (IntermediateField.comap i (separableClosure F K)) (separableClosure F E)
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
    ⊢ Iff (Membership.mem (IntermediateField.comap i (separableClosure F K)) x) (M …
  -/
  exact map_mem_separableClosure_iff i
  /-
    🎉 no goals
  -/


/-- If `i` is an `F`-algebra homomorphism from `E` to `K`, then the image of `separableClosure F E`
under the map `i` is contained in `separableClosure F K`. -/
theorem separableClosure.map_le_of_algHom (i : E →ₐ[F] K) :
    (separableClosure F E).map i ≤ separableClosure F K :=
  map_le_iff_le_comap.2 (comap_eq_of_algHom i).ge


variable (F) in
/-- If `K / E / F` is a field extension tower, such that `K / E` has no non-trivial separable
subextensions (when `K / E` is algebraic, this means that it is purely inseparable),
then the image of `separableClosure F E` in `K` is equal to `separableClosure F K`. -/
theorem separableClosure.map_eq_of_separableClosure_eq_bot [Algebra E K] [IsScalarTower F E K]
    (h : separableClosure E K = ⊥) :
    (separableClosure F E).map (IsScalarTower.toAlgHom F E K) = separableClosure F K := by
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
    h : Eq (separableClosure E K) Bot.bot
    ⊢ Eq (IntermediateField.map (IsScalarTower.toAlgHom F E K) (separableClosure F …
  -/
  refine le_antisymm (map_le_of_algHom _) (fun x hx ↦ ?_)
  obtain ⟨y, rfl⟩ := mem_bot.1 <| h ▸ mem_separableClosure_iff.2
    (IsSeparable.tower_top E <| mem_separableClosure_iff.1 hx)
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
    h : Eq (separableClosure E K) Bot.bot
    y : E
    hx : Membership.mem (separableClosure F K) ((algebraMap E K) y)
    ⊢ Membership.mem (IntermediateField.map (IsScalarTower.toAlgHom F E K) (separa …
  -/
  exact ⟨y, (map_mem_separableClosure_iff <| IsScalarTower.toAlgHom F E K).mp hx, rfl⟩
  /-
    🎉 no goals
  -/


/-- If `i` is an `F`-algebra isomorphism of `E` and `K`, then the image of `separableClosure F E`
under the map `i` is equal to `separableClosure F K`. -/
theorem separableClosure.map_eq_of_algEquiv (i : E ≃ₐ[F] K) :
    (separableClosure F E).map i = separableClosure F K :=
  (map_le_of_algHom i.toAlgHom).antisymm
                                                                 /-
                                                                   F : Type u
                                                                   E : Type v
                                                                   inst✝⁴ : Field F
                                                                   inst✝³ : Field E
                                                                   inst✝² : Algebra F E
                                                                   K : Type w
                                                                   inst✝¹ : Field K
                                                                   inst✝ : Algebra F K
                                                                   i : AlgEquiv F E K
                                                                   x : K
                                                                   h : Membership.mem (separableClosure F K) x
                                                                   ⊢ Eq (↑↑i (↑i.symm x)) x
                                                                 -/
    (fun x h ↦ ⟨_, (map_mem_separableClosure_iff i.symm).2 h, by simp⟩)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- If `E` and `K` are isomorphic as `F`-algebras, then `separableClosure F E` and
`separableClosure F K` are also isomorphic as `F`-algebras. -/
def separableClosure.algEquivOfAlgEquiv (i : E ≃ₐ[F] K) :
    separableClosure F E ≃ₐ[F] separableClosure F K :=
  (intermediateFieldMap i _).trans (equivOfEq (map_eq_of_algEquiv i))


alias AlgEquiv.separableClosure := separableClosure.algEquivOfAlgEquiv


/-- The separable closure of `F` in `E` is algebraic over `F`. -/
instance separableClosure.isAlgebraic : Algebra.IsAlgebraic F (separableClosure F E) :=
  ⟨fun x ↦ isAlgebraic_iff.2 (IsSeparable.isIntegral x.2).isAlgebraic⟩


/-- The separable closure of `F` in `E` is separable over `F`. -/
@[stacks 030K "$E_{sep}/F$ is separable"]
instance separableClosure.isSeparable : Algebra.IsSeparable F (separableClosure F E) :=
              /-
                F : Type u
                E : Type v
                inst✝⁴ : Field F
                inst✝³ : Field E
                inst✝² : Algebra F E
                K : Type w
                inst✝¹ : Field K
                inst✝ : Algebra F K
                x : Subtype fun x => Membership.mem (separableClosure F E) x
                ⊢ IsSeparable F x
              -/
  ⟨fun x ↦ by simpa only [IsSeparable, minpoly_eq] using x.2⟩
              /-
                🎉 no goals
              -/


/-- An intermediate field of `E / F` is contained in the separable closure of `F` in `E`
if all of its elements are separable over `F`. -/
theorem le_separableClosure' {L : IntermediateField F E} (hs : ∀ x : L, IsSeparable F x) :
                                             /-
                                               F : Type u
                                               E : Type v
                                               inst✝² : Field F
                                               inst✝¹ : Field E
                                               inst✝ : Algebra F E
                                               L : IntermediateField F E
                                               hs : ∀ (x : Subtype fun x => Membership.mem L x), IsSeparable F x
                                               x : E
                                               h : Membership.mem L x
                                               ⊢ Membership.mem (separableClosure F E) x
                                             -/
    L ≤ separableClosure F E := fun x h ↦ by simpa only [IsSeparable, minpoly_eq] using hs ⟨x, h⟩
                                             /-
                                               🎉 no goals
                                             -/


/-- An intermediate field of `E / F` is contained in the separable closure of `F` in `E`
if it is separable over `F`. -/
theorem le_separableClosure (L : IntermediateField F E) [Algebra.IsSeparable F L] :
    L ≤ separableClosure F E := le_separableClosure' F E (Algebra.IsSeparable.isSeparable F)


/-- An intermediate field of `E / F` is contained in the separable closure of `F` in `E`
if and only if it is separable over `F`. -/
theorem le_separableClosure_iff (L : IntermediateField F E) :
    L ≤ separableClosure F E ↔ Algebra.IsSeparable F L :=
                       /-
                         F : Type u
                         E : Type v
                         inst✝² : Field F
                         inst✝¹ : Field E
                         inst✝ : Algebra F E
                         L : IntermediateField F E
                         h : LE.le L (separableClosure F E)
                         x : Subtype fun x => Membership.mem L x
                         ⊢ IsSeparable F x
                       -/
  ⟨fun h ↦ ⟨fun x ↦ by simpa only [IsSeparable, minpoly_eq] using h x.2⟩,
                       /-
                         🎉 no goals
                       -/
    fun _ ↦ le_separableClosure _ _ _⟩


/-- The separable closure in `E` of the separable closure of `F` in `E` is equal to itself. -/
theorem separableClosure.separableClosure_eq_bot :
    separableClosure (separableClosure F E) E = ⊥ :=
  bot_unique fun x hx ↦ mem_bot.2
    ⟨⟨x, IsSeparable.of_algebra_isSeparable_of_isSeparable F (mem_separableClosure_iff.1 hx)⟩, rfl⟩


/-- The normal closure in `E/F` of the separable closure of `F` in `E` is equal to itself. -/
theorem separableClosure.normalClosure_eq_self :
    normalClosure F (separableClosure F E) E = separableClosure F E :=
  le_antisymm (normalClosure_le_iff.2 fun i ↦
    have : Algebra.IsSeparable F i.fieldRange :=
      (AlgEquiv.Algebra.isSeparable (AlgEquiv.ofInjectiveField i))
    le_separableClosure F E _) (le_normalClosure _)


/-- If `E` is normal over `F`, then the separable closure of `F` in `E` is Galois (i.e.
normal and separable) over `F`. -/
@[stacks 0EXK]
instance separableClosure.isGalois [Normal F E] : IsGalois F (separableClosure F E) where
  to_isSeparable := separableClosure.isSeparable F E
  to_normal := by
    /-
      F : Type u
      E : Type v
      inst✝⁵ : Field F
      inst✝⁴ : Field E
      inst✝³ : Algebra F E
      K : Type w
      inst✝² : Field K
      inst✝¹ : Algebra F K
      inst✝ : Normal F E
      ⊢ Normal F (Subtype fun x => Membership.mem (separableClosure F E) x)
    -/
    rw [← separableClosure.normalClosure_eq_self]
    /-
      F : Type u
      E : Type v
      inst✝⁵ : Field F
      inst✝⁴ : Field E
      inst✝³ : Algebra F E
      K : Type w
      inst✝² : Field K
      inst✝¹ : Algebra F K
      inst✝ : Normal F E
      ⊢ Normal F (Subtype fun x => Membership.mem (normalClosure F (Subtype fun x => …
    -/
    exact normalClosure.normal F _ E
    /-
      🎉 no goals
    -/


/-- If `E / F` is a field extension and `E` is separably closed, then the separable closure
of `F` in `E` is equal to `F` if and only if `F` is separably closed. -/
theorem IsSepClosed.separableClosure_eq_bot_iff [IsSepClosed E] :
    separableClosure F E = ⊥ ↔ IsSepClosed F := by
  refine ⟨fun h ↦ IsSepClosed.of_exists_root _ fun p _ hirr hsep ↦ ?_,
    fun _ ↦ IntermediateField.eq_bot_of_isSepClosed_of_isSeparable _⟩
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : IsSepClosed E
    h : Eq (separableClosure F E) Bot.bot
    p : Polynomial F
    x✝ : p.Monic
    hirr : Irreducible p
    hsep : p.Separable
    ⊢ Exists fun x => Eq (Polynomial.eval x p) 0
  -/
  obtain ⟨x, hx⟩ := IsSepClosed.exists_aeval_eq_zero E p (degree_pos_of_irreducible hirr).ne' hsep
  /-
    case intro
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : IsSepClosed E
    h : Eq (separableClosure F E) Bot.bot
    p : Polynomial F
    x✝ : p.Monic
    hirr : Irreducible p
    hsep : p.Separable
    x : E
    hx : Eq ((Polynomial.aeval x) p) 0
    ⊢ Exists fun x => Eq (Polynomial.eval x p) 0
  -/
  obtain ⟨x, rfl⟩ := h ▸ mem_separableClosure_iff.2 (hsep.of_dvd <| minpoly.dvd _ x hx)
  /-
    case intro.intro
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : IsSepClosed E
    h : Eq (separableClosure F E) Bot.bot
    p : Polynomial F
    x✝ : p.Monic
    hirr : Irreducible p
    hsep : p.Separable
    x : F
    hx : Eq ((Polynomial.aeval ((Algebra.ofId F E).toRingHom x)) p) 0
    ⊢ Exists fun x => Eq (Polynomial.eval x p) 0
  -/
  exact ⟨x, by simpa [Algebra.ofId_apply] using hx⟩
  /-
    🎉 no goals
  -/


/-- If `E` is separably closed, then the separable closure of `F` in `E` is an absolute
separable closure of `F`. -/
instance separableClosure.isSepClosure [IsSepClosed E] : IsSepClosure F (separableClosure F E) :=
  ⟨(IsSepClosed.separableClosure_eq_bot_iff _ E).mp (separableClosure.separableClosure_eq_bot F E),
    isSeparable F E⟩


/-- The absolute separable closure is defined to be the relative separable closure inside the
algebraic closure. It is indeed a separable closure (`IsSepClosure`) by
`separableClosure.isSepClosure`, and it is Galois (`IsGalois`) by `separableClosure.isGalois`
or `IsSepClosure.isGalois`, and every separable extension embeds into it (`IsSepClosed.lift`). -/
abbrev SeparableClosure : Type _ := separableClosure F (AlgebraicClosure F)


/-- `F(S) / F` is a separable extension if and only if all elements of `S` are
separable elements. -/
theorem IntermediateField.isSeparable_adjoin_iff_isSeparable {S : Set E} :
    Algebra.IsSeparable F (adjoin F S) ↔ ∀ x ∈ S, IsSeparable F x :=
  (le_separableClosure_iff F E _).symm.trans adjoin_le_iff


/-- The separable closure of `F` in `E` is equal to `E` if and only if `E / F` is
separable. -/
theorem separableClosure.eq_top_iff : separableClosure F E = ⊤ ↔ Algebra.IsSeparable F E :=
  ⟨fun h ↦ ⟨fun _ ↦ mem_separableClosure_iff.1 (h ▸ mem_top)⟩,
    fun _ ↦ top_unique fun x _ ↦ mem_separableClosure_iff.2 (Algebra.IsSeparable.isSeparable _ x)⟩


/-- If `K / E / F` is a field extension tower, then `separableClosure F K` is contained in
`separableClosure E K`. -/
theorem separableClosure.le_restrictScalars [Algebra E K] [IsScalarTower F E K] :
    separableClosure F K ≤ (separableClosure E K).restrictScalars F :=
  fun _ ↦ IsSeparable.tower_top E


/-- If `K / E / F` is a field extension tower, such that `E / F` is separable, then
`separableClosure F K` is equal to `separableClosure E K`. -/
theorem separableClosure.eq_restrictScalars_of_isSeparable [Algebra E K] [IsScalarTower F E K]
    [Algebra.IsSeparable F E] : separableClosure F K = (separableClosure E K).restrictScalars F :=
  (separableClosure.le_restrictScalars F E K).antisymm fun _ h ↦
    IsSeparable.of_algebra_isSeparable_of_isSeparable F h


/-- If `K / E / F` is a field extension tower, then `E` adjoin `separableClosure F K` is contained
in `separableClosure E K`. -/
theorem separableClosure.adjoin_le [Algebra E K] [IsScalarTower F E K] :
    adjoin E (separableClosure F K) ≤ separableClosure E K :=
  adjoin_le_iff.2 <| le_restrictScalars F E K


/-- A compositum of two separable extensions is separable. -/
instance IntermediateField.isSeparable_sup (L1 L2 : IntermediateField F E)
    [h1 : Algebra.IsSeparable F L1] [h2 : Algebra.IsSeparable F L2] :
    Algebra.IsSeparable F (L1 ⊔ L2 : IntermediateField F E) := by
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
    h1 : Algebra.IsSeparable F (Subtype fun x => Membership.mem L1 x)
    h2 : Algebra.IsSeparable F (Subtype fun x => Membership.mem L2 x)
    ⊢ Algebra.IsSeparable F (Subtype fun x => Membership.mem (Max.max L1 L2) x)
  -/
  rw [← le_separableClosure_iff] at h1 h2 ⊢
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
    h1 : LE.le L1 (separableClosure F E)
    h2 : LE.le L2 (separableClosure F E)
    ⊢ LE.le (Max.max L1 L2) (separableClosure F E)
  -/
  exact sup_le h1 h2
  /-
    🎉 no goals
  -/


/-- A compositum of separable extensions is separable. -/
instance IntermediateField.isSeparable_iSup {ι : Type*} {t : ι → IntermediateField F E}
    [h : ∀ i, Algebra.IsSeparable F (t i)] :
    Algebra.IsSeparable F (⨆ i, t i : IntermediateField F E) := by
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ι : Type u_1
    t : ι → IntermediateField F E
    h : ∀ (i : ι), Algebra.IsSeparable F (Subtype fun x => Membership.mem (t i) x)
    ⊢ Algebra.IsSeparable F (Subtype fun x => Membership.mem (iSup fun i => t i) x)
  -/
  simp_rw [← le_separableClosure_iff] at h ⊢
  /-
    F : Type u
    E : Type v
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Algebra F E
    K : Type w
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ι : Type u_1
    t : ι → IntermediateField F E
    h : ∀ (i : ι), LE.le (t i) (separableClosure F E)
    ⊢ LE.le (iSup fun i => t i) (separableClosure F E)
  -/
  exact iSup_le h
  /-
    🎉 no goals
  -/


/-- The (infinite) separable degree for a general field extension `E / F` is defined
to be the degree of `separableClosure F E / F`. -/
@[stacks 030L "Part 1"]
def sepDegree := Module.rank F (separableClosure F E)


/-- The (infinite) inseparable degree for a general field extension `E / F` is defined
to be the degree of `E / separableClosure F E`. -/
@[stacks 030L "Part 2"]
def insepDegree := Module.rank (separableClosure F E) E


/-- The finite inseparable degree for a general field extension `E / F` is defined
to be the degree of `E / separableClosure F E` as a natural number. It is defined to be zero
if such field extension is infinite. -/
def finInsepDegree : ℕ := finrank (separableClosure F E) E


theorem finInsepDegree_def' : finInsepDegree F E = Cardinal.toNat (insepDegree F E) := rfl


instance instNeZeroSepDegree : NeZero (sepDegree F E) := ⟨rank_pos.ne'⟩


instance instNeZeroInsepDegree : NeZero (insepDegree F E) := ⟨rank_pos.ne'⟩


instance instNeZeroFinInsepDegree [FiniteDimensional F E] :
    NeZero (finInsepDegree F E) := ⟨finrank_pos.ne'⟩


/-- If `E` and `K` are isomorphic as `F`-algebras, then they have the same
separable degree over `F`. -/
theorem lift_sepDegree_eq_of_equiv (i : E ≃ₐ[F] K) :
    Cardinal.lift.{w} (sepDegree F E) = Cardinal.lift.{v} (sepDegree F K) :=
  i.separableClosure.toLinearEquiv.lift_rank_eq


/-- The same-universe version of `Field.lift_sepDegree_eq_of_equiv`. -/
theorem sepDegree_eq_of_equiv (K : Type v) [Field K] [Algebra F K] (i : E ≃ₐ[F] K) :
    sepDegree F E = sepDegree F K :=
  i.separableClosure.toLinearEquiv.rank_eq


/-- The separable degree multiplied by the inseparable degree is equal
to the (infinite) field extension degree. -/
theorem sepDegree_mul_insepDegree : sepDegree F E * insepDegree F E = Module.rank F E :=
  rank_mul_rank F (separableClosure F E) E


/-- If `E` and `K` are isomorphic as `F`-algebras, then they have the same
inseparable degree over `F`. -/
theorem lift_insepDegree_eq_of_equiv (i : E ≃ₐ[F] K) :
    Cardinal.lift.{w} (insepDegree F E) = Cardinal.lift.{v} (insepDegree F K) :=
  Algebra.lift_rank_eq_of_equiv_equiv i.separableClosure i rfl


/-- The same-universe version of `Field.lift_insepDegree_eq_of_equiv`. -/
theorem insepDegree_eq_of_equiv (K : Type v) [Field K] [Algebra F K] (i : E ≃ₐ[F] K) :
    insepDegree F E = insepDegree F K :=
  Algebra.rank_eq_of_equiv_equiv i.separableClosure i rfl


/-- If `E` and `K` are isomorphic as `F`-algebras, then they have the same finite
inseparable degree over `F`. -/
theorem finInsepDegree_eq_of_equiv (i : E ≃ₐ[F] K) :
    finInsepDegree F E = finInsepDegree F K := by
  simpa only [Cardinal.toNat_lift] using congr_arg Cardinal.toNat
    (lift_insepDegree_eq_of_equiv F E K i)


@[simp]
theorem sepDegree_self : sepDegree F F = 1 := by
  /-
    F : Type u
    inst✝ : Field F
    ⊢ Eq (Field.sepDegree F F) 1
  -/
  rw [sepDegree, Subsingleton.elim (separableClosure F F) ⊥, IntermediateField.rank_bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem insepDegree_self : insepDegree F F = 1 := by
  /-
    F : Type u
    inst✝ : Field F
    ⊢ Eq (Field.insepDegree F F) 1
  -/
  rw [insepDegree, Subsingleton.elim (separableClosure F F) ⊤, IntermediateField.rank_top]
  /-
    🎉 no goals
  -/


@[simp]
theorem finInsepDegree_self : finInsepDegree F F = 1 := by
  /-
    F : Type u
    inst✝ : Field F
    ⊢ Eq (Field.finInsepDegree F F) 1
  -/
  rw [finInsepDegree_def', insepDegree_self, Cardinal.one_toNat]
  /-
    🎉 no goals
  -/


@[simp]
theorem sepDegree_bot : sepDegree F (⊥ : IntermediateField F E) = 1 := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Eq (Field.sepDegree F (Subtype fun x => Membership.mem Bot.bot x)) 1
  -/
  have := lift_sepDegree_eq_of_equiv _ _ _ (botEquiv F E)
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    this : Eq (Cardinal.lift.{u, v} (Field.sepDegree F (Subtype fun x => Membershi …
    ⊢ Eq (Field.sepDegree F (Subtype fun x => Membership.mem Bot.bot x)) 1
  -/
  rwa [sepDegree_self, Cardinal.lift_one, ← Cardinal.lift_one.{v, u}, Cardinal.lift_inj] at this
  /-
    🎉 no goals
  -/


@[simp]
theorem insepDegree_bot : insepDegree F (⊥ : IntermediateField F E) = 1 := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Eq (Field.insepDegree F (Subtype fun x => Membership.mem Bot.bot x)) 1
  -/
  have := lift_insepDegree_eq_of_equiv _ _ _ (botEquiv F E)
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    this : Eq (Cardinal.lift.{u, v} (Field.insepDegree F (Subtype fun x => Members …
    ⊢ Eq (Field.insepDegree F (Subtype fun x => Membership.mem Bot.bot x)) 1
  -/
  rwa [insepDegree_self, Cardinal.lift_one, ← Cardinal.lift_one.{v, u}, Cardinal.lift_inj] at this
  /-
    🎉 no goals
  -/


@[simp]
theorem finInsepDegree_bot : finInsepDegree F (⊥ : IntermediateField F E) = 1 := by
  /-
    F : Type u
    E : Type v
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Eq (Field.finInsepDegree F (Subtype fun x => Membership.mem Bot.bot x)) 1
  -/
  rw [finInsepDegree_eq_of_equiv _ _ _ (botEquiv F E), finInsepDegree_self]
  /-
    🎉 no goals
  -/


theorem lift_sepDegree_bot' : Cardinal.lift.{v} (sepDegree F (⊥ : IntermediateField E K)) =
    Cardinal.lift.{w} (sepDegree F E) :=
  lift_sepDegree_eq_of_equiv _ _ _ ((botEquiv E K).restrictScalars F)


theorem lift_insepDegree_bot' : Cardinal.lift.{v} (insepDegree F (⊥ : IntermediateField E K)) =
    Cardinal.lift.{w} (insepDegree F E) :=
  lift_insepDegree_eq_of_equiv _ _ _ ((botEquiv E K).restrictScalars F)


@[simp]
theorem finInsepDegree_bot' :
    finInsepDegree F (⊥ : IntermediateField E K) = finInsepDegree F E := by
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
    ⊢ Eq (Field.finInsepDegree F (Subtype fun x => Membership.mem Bot.bot x)) (Fie …
  -/
  simpa only [Cardinal.toNat_lift] using congr_arg Cardinal.toNat (lift_insepDegree_bot' F E K)
  /-
    🎉 no goals
  -/


@[simp]
theorem sepDegree_top : sepDegree F (⊤ : IntermediateField E K) = sepDegree F K :=
  sepDegree_eq_of_equiv _ _ _ ((topEquiv (F := E) (E := K)).restrictScalars F)


@[simp]
theorem insepDegree_top : insepDegree F (⊤ : IntermediateField E K) = insepDegree F K :=
  insepDegree_eq_of_equiv _ _ _ ((topEquiv (F := E) (E := K)).restrictScalars F)


@[simp]
theorem finInsepDegree_top : finInsepDegree F (⊤ : IntermediateField E K) = finInsepDegree F K := by
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
    ⊢ Eq (Field.finInsepDegree F (Subtype fun x => Membership.mem Top.top x)) (Fie …
  -/
  rw [finInsepDegree_def', insepDegree_top, ← finInsepDegree_def']
  /-
    🎉 no goals
  -/


@[simp]
theorem sepDegree_bot' : sepDegree F (⊥ : IntermediateField E K) = sepDegree F E :=
  sepDegree_eq_of_equiv _ _ _ ((botEquiv E K).restrictScalars F)


@[simp]
theorem insepDegree_bot' : insepDegree F (⊥ : IntermediateField E K) = insepDegree F E :=
  insepDegree_eq_of_equiv _ _ _ ((botEquiv E K).restrictScalars F)


/-- A separable extension has separable degree equal to degree. -/
theorem Algebra.IsSeparable.sepDegree_eq [Algebra.IsSeparable F E] :
    sepDegree F E = Module.rank F E := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsSeparable F E
    ⊢ Eq (Field.sepDegree F E) (Module.rank F E)
  -/
  rw [sepDegree, (separableClosure.eq_top_iff F E).2 ‹_›, IntermediateField.rank_top']
  /-
    🎉 no goals
  -/


/-- A separable extension has inseparable degree one. -/
theorem Algebra.IsSeparable.insepDegree_eq [Algebra.IsSeparable F E] : insepDegree F E = 1 := by
  /-
    F : Type u
    E : Type v
    inst✝³ : Field F
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : Algebra.IsSeparable F E
    ⊢ Eq (Field.insepDegree F E) 1
  -/
  rw [insepDegree, (separableClosure.eq_top_iff F E).2 ‹_›, IntermediateField.rank_top]
  /-
    🎉 no goals
  -/


/-- A separable extension has finite inseparable degree one. -/
theorem Algebra.IsSeparable.finInsepDegree_eq [Algebra.IsSeparable F E] : finInsepDegree F E = 1 :=
  Cardinal.one_toNat ▸ congr(Cardinal.toNat $(insepDegree_eq F E))

