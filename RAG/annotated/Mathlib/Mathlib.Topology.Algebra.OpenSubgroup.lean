/-- The type of open subgroups of a topological additive group. -/
structure OpenAddSubgroup (G : Type*) [AddGroup G] [TopologicalSpace G] extends AddSubgroup G where
  isOpen' : IsOpen carrier


/-- The type of open subgroups of a topological group. -/
@[to_additive]
structure OpenSubgroup (G : Type*) [Group G] [TopologicalSpace G] extends Subgroup G where
  isOpen' : IsOpen carrier


@[to_additive]
instance hasCoeSubgroup : CoeTC (OpenSubgroup G) (Subgroup G) :=
  ⟨toSubgroup⟩


@[to_additive]
theorem toSubgroup_injective : Injective ((↑) : OpenSubgroup G → Subgroup G)
  | ⟨_, _⟩, ⟨_, _⟩, rfl => rfl


@[to_additive]
instance : SetLike (OpenSubgroup G) G where
  coe U := U.1
  coe_injective' _ _ h := toSubgroup_injective <| SetLike.ext' h


@[to_additive]
instance : SubgroupClass (OpenSubgroup G) G where
  mul_mem := Subsemigroup.mul_mem' _
  one_mem U := U.one_mem'
  inv_mem := Subgroup.inv_mem' _


/-- Coercion from `OpenSubgroup G` to `Opens G`. -/
@[to_additive (attr := coe) "Coercion from `OpenAddSubgroup G` to `Opens G`."]
def toOpens (U : OpenSubgroup G) : Opens G := ⟨U, U.isOpen'⟩


@[to_additive]
instance hasCoeOpens : CoeTC (OpenSubgroup G) (Opens G) := ⟨toOpens⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_toOpens : ((U : Opens G) : Set G) = U :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_toSubgroup : ((U : Subgroup G) : Set G) = U := rfl


@[to_additive (attr := simp, norm_cast)]
theorem mem_toOpens : g ∈ (U : Opens G) ↔ g ∈ U := Iff.rfl


@[to_additive (attr := simp, norm_cast)]
theorem mem_toSubgroup : g ∈ (U : Subgroup G) ↔ g ∈ U := Iff.rfl


@[to_additive (attr := ext)]
theorem ext (h : ∀ x, x ∈ U ↔ x ∈ V) : U = V :=
  SetLike.ext h


@[to_additive]
protected theorem isOpen : IsOpen (U : Set G) :=
  U.isOpen'


@[to_additive]
theorem mem_nhds_one : (U : Set G) ∈ 𝓝 (1 : G) :=
  U.isOpen.mem_nhds U.one_mem


@[to_additive] instance : Top (OpenSubgroup G) := ⟨⟨⊤, isOpen_univ⟩⟩


@[to_additive (attr := simp)]
theorem mem_top (x : G) : x ∈ (⊤ : OpenSubgroup G) :=
  trivial


@[to_additive (attr := simp, norm_cast)]
theorem coe_top : ((⊤ : OpenSubgroup G) : Set G) = Set.univ :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem toSubgroup_top : ((⊤ : OpenSubgroup G) : Subgroup G) = ⊤ :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem toOpens_top : ((⊤ : OpenSubgroup G) : Opens G) = ⊤ :=
  rfl


@[to_additive]
instance : Inhabited (OpenSubgroup G) :=
  ⟨⊤⟩


@[to_additive]
theorem isClosed [ContinuousMul G] (U : OpenSubgroup G) : IsClosed (U : Set G) := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    U : OpenSubgroup G
    ⊢ IsClosed ↑U
  -/
  apply isOpen_compl_iff.1
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    U : OpenSubgroup G
    ⊢ IsOpen (HasCompl.compl ↑U)
  -/
  refine isOpen_iff_forall_mem_open.2 fun x hx ↦ ⟨(fun y ↦ y * x⁻¹) ⁻¹' U, ?_, ?_, ?_⟩
    /-
      case refine_1
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : ContinuousMul G
      U : OpenSubgroup G
      x : G
      hx : Membership.mem (HasCompl.compl ↑U) x
      ⊢ HasSubset.Subset (Set.preimage (fun y => HMul.hMul y (Inv.inv x)) ↑U) (HasCo …
    -/
  · refine fun u hux hu ↦ hx ?_
    /-
      case refine_1
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : ContinuousMul G
      U : OpenSubgroup G
      x : G
      hx : Membership.mem (HasCompl.compl ↑U) x
      u : G
      hux : Membership.mem (Set.preimage (fun y => HMul.hMul y (Inv.inv x)) ↑U) u
      hu : Membership.mem (↑U) u
      ⊢ Membership.mem (↑U) x
    -/
    simp only [Set.mem_preimage, SetLike.mem_coe] at hux hu ⊢
    /-
      case refine_1
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : ContinuousMul G
      U : OpenSubgroup G
      x : G
      hx : Membership.mem (HasCompl.compl ↑U) x
      u : G
      hux : Membership.mem U (HMul.hMul u (Inv.inv x))
      hu : Membership.mem U u
      ⊢ Membership.mem U x
    -/
    convert U.mul_mem (U.inv_mem hux) hu
    /-
      case h.e'_1
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : ContinuousMul G
      U : OpenSubgroup G
      x : G
      hx : Membership.mem (HasCompl.compl ↑U) x
      u : G
      hux : Membership.mem U (HMul.hMul u (Inv.inv x))
      hu : Membership.mem U u
      ⊢ Eq x (HMul.hMul (Inv.inv (HMul.hMul u (Inv.inv x))) u)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : ContinuousMul G
      U : OpenSubgroup G
      x : G
      hx : Membership.mem (HasCompl.compl ↑U) x
      ⊢ IsOpen (Set.preimage (fun y => HMul.hMul y (Inv.inv x)) ↑U)
    -/
  · exact U.isOpen.preimage (continuous_mul_right _)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : ContinuousMul G
      U : OpenSubgroup G
      x : G
      hx : Membership.mem (HasCompl.compl ↑U) x
      ⊢ Membership.mem (Set.preimage (fun y => HMul.hMul y (Inv.inv x)) ↑U) x
    -/
  · simp [one_mem]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem isClopen [ContinuousMul G] (U : OpenSubgroup G) : IsClopen (U : Set G) :=
  ⟨U.isClosed, U.isOpen⟩


/-- The product of two open subgroups as an open subgroup of the product group. -/
@[to_additive "The product of two open subgroups as an open subgroup of the product group."]
def prod (U : OpenSubgroup G) (V : OpenSubgroup H) : OpenSubgroup (G × H) :=
  ⟨.prod U V, U.isOpen.prod V.isOpen⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_prod (U : OpenSubgroup G) (V : OpenSubgroup H) :
    (U.prod V : Set (G × H)) = (U : Set G) ×ˢ (V : Set H) :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem toSubgroup_prod (U : OpenSubgroup G) (V : OpenSubgroup H) :
    (U.prod V : Subgroup (G × H)) = (U : Subgroup G).prod V :=
  rfl


@[to_additive]
instance instInfOpenSubgroup : Min (OpenSubgroup G) :=
  ⟨fun U V ↦ ⟨U ⊓ V, U.isOpen.inter V.isOpen⟩⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_inf : (↑(U ⊓ V) : Set G) = (U : Set G) ∩ V :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem toSubgroup_inf : (↑(U ⊓ V) : Subgroup G) = ↑U ⊓ ↑V :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem toOpens_inf : (↑(U ⊓ V) : Opens G) = ↑U ⊓ ↑V :=
  rfl


@[to_additive (attr := simp)]
theorem mem_inf {x} : x ∈ U ⊓ V ↔ x ∈ U ∧ x ∈ V :=
  Iff.rfl


@[to_additive]
instance instPartialOrderOpenSubgroup : PartialOrder (OpenSubgroup G) := inferInstance

-- Porting note: we override `toPartialorder` to get better `le`

@[to_additive]
instance instSemilatticeInfOpenSubgroup : SemilatticeInf (OpenSubgroup G) :=
  { SetLike.coe_injective.semilatticeInf ((↑) : OpenSubgroup G → Set G) fun _ _ ↦ rfl with
    toPartialOrder := instPartialOrderOpenSubgroup }


@[to_additive]
instance : OrderTop (OpenSubgroup G) where
  top := ⊤
  le_top _ := Set.subset_univ _


@[to_additive (attr := simp, norm_cast)]
theorem toSubgroup_le : (U : Subgroup G) ≤ (V : Subgroup G) ↔ U ≤ V :=
  Iff.rfl


/-- The preimage of an `OpenSubgroup` along a continuous `Monoid` homomorphism
  is an `OpenSubgroup`. -/
@[to_additive "The preimage of an `OpenAddSubgroup` along a continuous `AddMonoid` homomorphism
is an `OpenAddSubgroup`."]
def comap (f : G →* N) (hf : Continuous f) (H : OpenSubgroup N) : OpenSubgroup G :=
  ⟨.comap f H, H.isOpen.preimage hf⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_comap (H : OpenSubgroup N) (f : G →* N) (hf : Continuous f) :
    (H.comap f hf : Set G) = f ⁻¹' H :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem toSubgroup_comap (H : OpenSubgroup N) (f : G →* N) (hf : Continuous f) :
    (H.comap f hf : Subgroup G) = (H : Subgroup N).comap f :=
  rfl


@[to_additive (attr := simp)]
theorem mem_comap {H : OpenSubgroup N} {f : G →* N} {hf : Continuous f} {x : G} :
    x ∈ H.comap f hf ↔ f x ∈ H :=
  Iff.rfl


@[to_additive]
theorem comap_comap {P : Type*} [Group P] [TopologicalSpace P] (K : OpenSubgroup P) (f₂ : N →* P)
    (hf₂ : Continuous f₂) (f₁ : G →* N) (hf₁ : Continuous f₁) :
    (K.comap f₂ hf₂).comap f₁ hf₁ = K.comap (f₂.comp f₁) (hf₂.comp hf₁) :=
  rfl


@[to_additive]
theorem isOpen_of_mem_nhds [ContinuousMul G] (H : Subgroup G) {g : G} (hg : (H : Set G) ∈ 𝓝 g) :
    IsOpen (H : Set G) := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    H : Subgroup G
    g : G
    hg : Membership.mem (nhds g) ↑H
    ⊢ IsOpen ↑H
  -/
  refine isOpen_iff_mem_nhds.2 fun x hx ↦ ?_
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    H : Subgroup G
    g : G
    hg : Membership.mem (nhds g) ↑H
    x : G
    hx : Membership.mem (↑H) x
    ⊢ Membership.mem (nhds x) ↑H
  -/
  have hg' : g ∈ H := SetLike.mem_coe.1 (mem_of_mem_nhds hg)
  have : Filter.Tendsto (fun y ↦ y * (x⁻¹ * g)) (𝓝 x) (𝓝 g) :=
    (continuous_id.mul continuous_const).tendsto' _ _ (mul_inv_cancel_left _ _)
  simpa only [SetLike.mem_coe, Filter.mem_map',
    H.mul_mem_cancel_right (H.mul_mem (H.inv_mem hx) hg')] using this hg


@[to_additive]
theorem isOpen_mono [ContinuousMul G] {H₁ H₂ : Subgroup G} (h : H₁ ≤ H₂)
    (h₁ : IsOpen (H₁ : Set G)) : IsOpen (H₂ : Set G) :=
  isOpen_of_mem_nhds _ <| Filter.mem_of_superset (h₁.mem_nhds <| one_mem H₁) h


@[to_additive]
theorem isOpen_of_openSubgroup [ContinuousMul G] (H: Subgroup G) {U : OpenSubgroup G} (h : ↑U ≤ H) :
    IsOpen (H : Set G) :=
  isOpen_mono h U.isOpen


/-- If a subgroup of a topological group has `1` in its interior, then it is open. -/
@[to_additive "If a subgroup of an additive topological group has `0` in its interior, then it is
open."]
theorem isOpen_of_one_mem_interior [ContinuousMul G] (H: Subgroup G)
    (h_1_int : (1 : G) ∈ interior (H : Set G)) : IsOpen (H : Set G) :=
  isOpen_of_mem_nhds H <| mem_interior_iff_mem_nhds.1 h_1_int


@[to_additive]
lemma isClosed_of_isOpen [ContinuousMul G] (U : Subgroup G) (h : IsOpen (U : Set G)) :
    IsClosed (U : Set G) :=
  OpenSubgroup.isClosed ⟨U, h⟩


@[to_additive]
lemma subgroupOf_isOpen (U K : Subgroup G) (h : IsOpen (K : Set G)) :
    IsOpen (K.subgroupOf U : Set U) :=
  Continuous.isOpen_preimage (continuous_iff_le_induced.mpr fun _ ↦ id) _ h


@[to_additive]
lemma discreteTopology [ContinuousMul G] (U : Subgroup G) (h : IsOpen (U : Set G)) :
    DiscreteTopology (G ⧸ U) := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    U : Subgroup G
    h : IsOpen ↑U
    ⊢ DiscreteTopology (HasQuotient.Quotient G U)
  -/
  refine singletons_open_iff_discrete.mp (fun g ↦ ?_)
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    U : Subgroup G
    h : IsOpen ↑U
    g : HasQuotient.Quotient G U
    ⊢ IsOpen (Singleton.singleton g)
  -/
  induction' g using Quotient.inductionOn with g
  /-
    case h
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    U : Subgroup G
    h : IsOpen ↑U
    g : G
    ⊢ IsOpen (Singleton.singleton (Quotient.mk (QuotientGroup.leftRel U) g))
  -/
  show IsOpen (QuotientGroup.mk ⁻¹' {QuotientGroup.mk g})
  /-
    case h
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousMul G
    U : Subgroup G
    h : IsOpen ↑U
    g : G
    ⊢ IsOpen (Set.preimage QuotientGroup.mk (Singleton.singleton ↑g))
  -/
  convert_to IsOpen ((g * ·) '' U)
    /-
      case h.e'_3
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : ContinuousMul G
      U : Subgroup G
      h : IsOpen ↑U
      g : G
      ⊢ Eq (Set.preimage QuotientGroup.mk (Singleton.singleton ↑g)) (Set.image (fun  …
    -/
  · ext g'
    /-
      case h.e'_3.h
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : ContinuousMul G
      U : Subgroup G
      h : IsOpen ↑U
      g g' : G
      ⊢ Iff (Membership.mem (Set.preimage QuotientGroup.mk (Singleton.singleton ↑g)) …
    -/
    simp only [Set.mem_preimage, Set.mem_singleton_iff, QuotientGroup.eq, Set.image_mul_left]
    /-
      case h.e'_3.h
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : ContinuousMul G
      U : Subgroup G
      h : IsOpen ↑U
      g g' : G
      ⊢ Iff (Membership.mem U (HMul.hMul (Inv.inv g') g)) (Membership.mem (↑U) (HMul …
    -/
    rw [← U.inv_mem_iff]
    /-
      case h.e'_3.h
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : ContinuousMul G
      U : Subgroup G
      h : IsOpen ↑U
      g g' : G
      ⊢ Iff (Membership.mem U (Inv.inv (HMul.hMul (Inv.inv g') g))) (Membership.mem  …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case h
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : TopologicalSpace G
      inst✝ : ContinuousMul G
      U : Subgroup G
      h : IsOpen ↑U
      g : G
      ⊢ IsOpen (Set.image (fun x => HMul.hMul g x) ↑U)
    -/
  · exact Homeomorph.mulLeft g |>.isOpen_image |>.mpr h
    /-
      🎉 no goals
    -/


@[to_additive]
instance [ContinuousMul G] (U : OpenSubgroup G) : DiscreteTopology (G ⧸ U.toSubgroup) :=
  discreteTopology U.toSubgroup U.isOpen


@[to_additive]
lemma quotient_finite_of_isOpen [ContinuousMul G] [CompactSpace G] (U : Subgroup G)
    (h : IsOpen (U : Set G)) : Finite (G ⧸ U) :=
  have : DiscreteTopology (G ⧸ U) := U.discreteTopology h
  finite_of_compact_of_discrete


@[to_additive]
instance [ContinuousMul G] [CompactSpace G] (U : OpenSubgroup G) : Finite (G ⧸ U.toSubgroup) :=
  quotient_finite_of_isOpen U.toSubgroup U.isOpen


@[to_additive]
lemma quotient_finite_of_isOpen' [TopologicalGroup G] [CompactSpace G] (U : Subgroup G)
    (K : Subgroup U) (hUopen : IsOpen (U : Set G)) (hKopen : IsOpen (K : Set U)) :
    Finite (U ⧸ K) :=
  have : CompactSpace U := isCompact_iff_compactSpace.mp <| IsClosed.isCompact <|
    U.isClosed_of_isOpen hUopen
  K.quotient_finite_of_isOpen hKopen


@[to_additive]
instance [TopologicalGroup G] [CompactSpace G] (U : OpenSubgroup G) (K : OpenSubgroup U) :
    Finite (U ⧸ K.toSubgroup) :=
  quotient_finite_of_isOpen' U.toSubgroup K.toSubgroup U.isOpen K.isOpen


@[to_additive]
instance : Max (OpenSubgroup G) :=
  ⟨fun U V ↦ ⟨U ⊔ V, Subgroup.isOpen_mono (le_sup_left : U.1 ≤ U.1 ⊔ V.1) U.isOpen⟩⟩


@[to_additive (attr := simp, norm_cast)]
theorem toSubgroup_sup (U V : OpenSubgroup G) : (↑(U ⊔ V) : Subgroup G) = ↑U ⊔ ↑V := rfl

-- Porting note: we override `toPartialorder` to get better `le`

@[to_additive]
instance : Lattice (OpenSubgroup G) :=
  { instSemilatticeInfOpenSubgroup,
    toSubgroup_injective.semilatticeSup ((↑) : OpenSubgroup G → Subgroup G) fun _ _ ↦ rfl with
    toPartialOrder := instPartialOrderOpenSubgroup }


theorem isOpen_mono {U P : Submodule R M} (h : U ≤ P) (hU : IsOpen (U : Set M)) :
    IsOpen (P : Set M) :=
  @AddSubgroup.isOpen_mono M _ _ _ U.toAddSubgroup P.toAddSubgroup h hU


theorem isOpen_of_isOpen_subideal {U I : Ideal R} (h : U ≤ I) (hU : IsOpen (U : Set R)) :
    IsOpen (I : Set R) :=
  @Submodule.isOpen_mono R R _ _ _ _ Semiring.toModule _ _ h hU


/-- The type of open normal subgroups of a topological group. -/
@[ext]
structure OpenNormalSubgroup (G : Type u) [Group G] [TopologicalSpace G]
  extends OpenSubgroup G where
  isNormal' : toSubgroup.Normal := by infer_instance


/-- The type of open normal subgroups of a topological additive group. -/
@[ext]
structure OpenNormalAddSubgroup (G : Type u) [AddGroup G] [TopologicalSpace G]
  extends OpenAddSubgroup G where
  isNormal' : toAddSubgroup.Normal := by infer_instance


@[to_additive]
instance (H : OpenNormalSubgroup G) : H.toSubgroup.Normal := H.isNormal'


@[to_additive]
theorem toSubgroup_injective : Function.Injective
    (fun H ↦ H.toOpenSubgroup.toSubgroup : OpenNormalSubgroup G → Subgroup G) :=
  fun A B h ↦ by
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : TopologicalSpace G
    A B : OpenNormalSubgroup G
    h : Eq ((fun H => ↑H.toOpenSubgroup) A) ((fun H => ↑H.toOpenSubgroup) B)
    ⊢ Eq A B
  -/
  ext
  /-
    case carrier.h
    G : Type u
    inst✝¹ : Group G
    inst✝ : TopologicalSpace G
    A B : OpenNormalSubgroup G
    h : Eq ((fun H => ↑H.toOpenSubgroup) A) ((fun H => ↑H.toOpenSubgroup) B)
    x✝ : G
    ⊢ Iff (Membership.mem (↑A.toOpenSubgroup).carrier x✝) (Membership.mem (↑B.toOp …
  -/
  dsimp at h
  /-
    case carrier.h
    G : Type u
    inst✝¹ : Group G
    inst✝ : TopologicalSpace G
    A B : OpenNormalSubgroup G
    h : Eq ↑A.toOpenSubgroup ↑B.toOpenSubgroup
    x✝ : G
    ⊢ Iff (Membership.mem (↑A.toOpenSubgroup).carrier x✝) (Membership.mem (↑B.toOp …
  -/
  rw [h]
  /-
    🎉 no goals
  -/


@[to_additive]
instance : SetLike (OpenNormalSubgroup G) G where
  coe U := U.1
  coe_injective' _ _ h := toSubgroup_injective <| SetLike.ext' h


@[to_additive]
instance : SubgroupClass (OpenNormalSubgroup G) G where
  mul_mem := Subsemigroup.mul_mem' _
  one_mem U := U.one_mem'
  inv_mem := Subgroup.inv_mem' _


@[to_additive]
instance : Coe (OpenNormalSubgroup G) (Subgroup G) where
  coe H := H.toOpenSubgroup.toSubgroup


@[to_additive]
instance instPartialOrderOpenNormalSubgroup : PartialOrder (OpenNormalSubgroup G) := inferInstance


@[to_additive]
instance instInfOpenNormalSubgroup : Min (OpenNormalSubgroup G) :=
  ⟨fun U V ↦ ⟨U.toOpenSubgroup ⊓ V.toOpenSubgroup,
    Subgroup.normal_inf_normal U.toSubgroup V.toSubgroup⟩⟩


@[to_additive]
instance instSemilatticeInfOpenNormalSubgroup : SemilatticeInf (OpenNormalSubgroup G) :=
  SetLike.coe_injective.semilatticeInf ((↑) : OpenNormalSubgroup G → Set G) fun _ _ ↦ rfl


@[to_additive]
instance [ContinuousMul G] : Max (OpenNormalSubgroup G) :=
  ⟨fun U V ↦ ⟨U.toOpenSubgroup ⊔ V.toOpenSubgroup,
    Subgroup.sup_normal U.toOpenSubgroup.1 V.toOpenSubgroup.1⟩⟩


@[to_additive]
instance instSemilatticeSupOpenNormalSubgroup [ContinuousMul G] :
    SemilatticeSup (OpenNormalSubgroup G) :=
  toSubgroup_injective.semilatticeSup _ (fun _ _ ↦ rfl)


@[to_additive]
instance [ContinuousMul G] : Lattice (OpenNormalSubgroup G) :=
  { instSemilatticeInfOpenNormalSubgroup,
    instSemilatticeSupOpenNormalSubgroup with
    toPartialOrder := instPartialOrderOpenNormalSubgroup}


structure TopologicalAddGroup.addNegClosureNhd (T W : Set G) [AddGroup G] : Prop where
  nhd : T ∈ 𝓝 0
  neg : -T = T
  isOpen : IsOpen T
  add : W + T ⊆ W


/-- For a set `W`, `T` is a neighborhood of `1` which is open, statble under inverse and satisfies
`T * W ⊆ W`. -/
@[to_additive
"For a set `W`, `T` is a neighborhood of `0` which is open, stable under negation and satisfies
`T + W ⊆ W`. "]
structure TopologicalGroup.mulInvClosureNhd (T W : Set G) [Group G] : Prop where
  nhd : T ∈ 𝓝 1
  inv : T⁻¹ = T
  isOpen : IsOpen T
  mul : W * T ⊆ W


@[to_additive]
lemma exist_mul_closure_nhd {W : Set G} (WClopen : IsClopen W) : ∃ T ∈ 𝓝 (1 : G), W * T ⊆ W := by
  apply WClopen.isClosed.isCompact.induction_on (p := fun S ↦ ∃ T ∈ 𝓝 (1 : G), S * T ⊆ W)
    ⟨Set.univ ,by simp only [univ_mem, empty_mul, empty_subset, and_self]⟩
    (fun _ _ huv ⟨T, hT, mem⟩ ↦ ⟨T, hT, (mul_subset_mul_right huv).trans mem⟩)
    fun U V ⟨T₁, hT₁, mem1⟩ ⟨T₂, hT₂, mem2⟩ ↦ ⟨T₁ ∩ T₂, inter_mem hT₁ hT₂, by
      rw [union_mul]
      exact union_subset (mul_subset_mul_left inter_subset_left |>.trans mem1)
        (mul_subset_mul_left inter_subset_right |>.trans mem2) ⟩
  /-
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    ⊢ ∀ (x : G), Membership.mem W x → Exists fun t => And (Membership.mem (nhdsWit …
  -/
  intro x memW
  /-
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    x : G
    memW : Membership.mem W x
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x W) t) (Exists fun T => And …
  -/
  have : (x, 1) ∈ (fun p ↦ p.1 * p.2) ⁻¹' W := by simp [memW]
  rcases isOpen_prod_iff.mp (continuous_mul.isOpen_preimage W <| WClopen.2) x 1 this with
    ⟨U, V, Uopen, Vopen, xmemU, onememV, prodsub⟩
  /-
    case intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    x : G
    memW : Membership.mem W x
    this : Membership.mem (Set.preimage (fun p => HMul.hMul p.1 p.2) W) { fst := x …
    U V : Set G
    Uopen : IsOpen U
    Vopen : IsOpen V
    xmemU : Membership.mem U x
    onememV : Membership.mem V 1
    prodsub : HasSubset.Subset (SProd.sprod U V) (Set.preimage (fun p => HMul.hMul …
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x W) t) (Exists fun T => And …
  -/
  have h6 : U * V ⊆ W := mul_subset_iff.mpr (fun _ hx _ hy ↦ prodsub (mk_mem_prod hx hy))
  exact ⟨U ∩ W, ⟨U, Uopen.mem_nhds xmemU, W, fun _ a ↦ a, rfl⟩,
    V, IsOpen.mem_nhds Vopen onememV, fun _ a ↦ h6 ((mul_subset_mul_right inter_subset_left) a)⟩


@[to_additive]
lemma exists_mulInvClosureNhd {W : Set G} (WClopen : IsClopen W) :
    ∃ T, mulInvClosureNhd T W := by
  /-
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    ⊢ Exists fun T => TopologicalGroup.mulInvClosureNhd T W
  -/
  rcases exist_mul_closure_nhd WClopen with ⟨S, Smemnhds, mulclose⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    S : Set G
    Smemnhds : Membership.mem (nhds 1) S
    mulclose : HasSubset.Subset (HMul.hMul W S) W
    ⊢ Exists fun T => TopologicalGroup.mulInvClosureNhd T W
  -/
  rcases mem_nhds_iff.mp Smemnhds with ⟨U, UsubS, Uopen, onememU⟩
  /-
    case intro.intro.intro.intro.intro
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    S : Set G
    Smemnhds : Membership.mem (nhds 1) S
    mulclose : HasSubset.Subset (HMul.hMul W S) W
    U : Set G
    UsubS : HasSubset.Subset U S
    Uopen : IsOpen U
    onememU : Membership.mem U 1
    ⊢ Exists fun T => TopologicalGroup.mulInvClosureNhd T W
  -/
  use U ∩ U⁻¹
  /-
    case h
    G : Type u_1
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    S : Set G
    Smemnhds : Membership.mem (nhds 1) S
    mulclose : HasSubset.Subset (HMul.hMul W S) W
    U : Set G
    UsubS : HasSubset.Subset U S
    Uopen : IsOpen U
    onememU : Membership.mem U 1
    ⊢ TopologicalGroup.mulInvClosureNhd (Inter.inter U (Inv.inv U)) W
  -/
  constructor
    /-
      case h.nhd
      G : Type u_1
      inst✝³ : TopologicalSpace G
      inst✝² : Group G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      W : Set G
      WClopen : IsClopen W
      S : Set G
      Smemnhds : Membership.mem (nhds 1) S
      mulclose : HasSubset.Subset (HMul.hMul W S) W
      U : Set G
      UsubS : HasSubset.Subset U S
      Uopen : IsOpen U
      onememU : Membership.mem U 1
      ⊢ Membership.mem (nhds 1) (Inter.inter U (Inv.inv U))
    -/
  · simp [Uopen.mem_nhds onememU, inv_mem_nhds_one]
    /-
      🎉 no goals
    -/
    /-
      case h.inv
      G : Type u_1
      inst✝³ : TopologicalSpace G
      inst✝² : Group G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      W : Set G
      WClopen : IsClopen W
      S : Set G
      Smemnhds : Membership.mem (nhds 1) S
      mulclose : HasSubset.Subset (HMul.hMul W S) W
      U : Set G
      UsubS : HasSubset.Subset U S
      Uopen : IsOpen U
      onememU : Membership.mem U 1
      ⊢ Eq (Inv.inv (Inter.inter U (Inv.inv U))) (Inter.inter U (Inv.inv U))
    -/
  · simp [inter_comm]
    /-
      🎉 no goals
    -/
    /-
      case h.isOpen
      G : Type u_1
      inst✝³ : TopologicalSpace G
      inst✝² : Group G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      W : Set G
      WClopen : IsClopen W
      S : Set G
      Smemnhds : Membership.mem (nhds 1) S
      mulclose : HasSubset.Subset (HMul.hMul W S) W
      U : Set G
      UsubS : HasSubset.Subset U S
      Uopen : IsOpen U
      onememU : Membership.mem U 1
      ⊢ IsOpen (Inter.inter U (Inv.inv U))
    -/
  · exact Uopen.inter Uopen.inv
    /-
      🎉 no goals
    -/
    /-
      case h.mul
      G : Type u_1
      inst✝³ : TopologicalSpace G
      inst✝² : Group G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      W : Set G
      WClopen : IsClopen W
      S : Set G
      Smemnhds : Membership.mem (nhds 1) S
      mulclose : HasSubset.Subset (HMul.hMul W S) W
      U : Set G
      UsubS : HasSubset.Subset U S
      Uopen : IsOpen U
      onememU : Membership.mem U 1
      ⊢ HasSubset.Subset (HMul.hMul W (Inter.inter U (Inv.inv U))) W
    -/
  · exact fun a ha ↦ mulclose (mul_subset_mul_left UsubS (mul_subset_mul_left inter_subset_left ha))
    /-
      🎉 no goals
    -/


@[to_additive]
theorem exist_openSubgroup_sub_clopen_nhd_of_one {G : Type*} [Group G] [TopologicalSpace G]
    [TopologicalGroup G] [CompactSpace G] {W : Set G} (WClopen : IsClopen W) (einW : 1 ∈ W) :
    ∃ H : OpenSubgroup G, (H : Set G) ⊆ W := by
  /-
    G : Type u_2
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    einW : Membership.mem W 1
    ⊢ Exists fun H => HasSubset.Subset (↑H) W
  -/
  rcases exists_mulInvClosureNhd WClopen with ⟨V, hV⟩
  let S : Subgroup G := {
    carrier := ⋃ n , V ^ (n + 1)
    mul_mem' := fun ha hb ↦ by
      rcases mem_iUnion.mp ha with ⟨k, hk⟩
      rcases mem_iUnion.mp hb with ⟨l, hl⟩
      apply mem_iUnion.mpr
      use k + 1 + l
      rw [add_assoc, pow_add]
      exact Set.mul_mem_mul hk hl
    one_mem' := by
      apply mem_iUnion.mpr
      use 0
      simp [mem_of_mem_nhds hV.nhd]
    inv_mem' := fun ha ↦ by
      rcases mem_iUnion.mp ha with ⟨k, hk⟩
      apply mem_iUnion.mpr
      use k
      rw [← hV.inv]
      simpa only [inv_pow, Set.mem_inv, inv_inv] using hk }
  have : IsOpen (⋃ n , V ^ (n + 1)) := by
    refine isOpen_iUnion (fun n ↦ ?_)
    rw [pow_succ]
    exact hV.isOpen.mul_left
  /-
    case intro
    G : Type u_2
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    einW : Membership.mem W 1
    V : Set G
    hV : TopologicalGroup.mulInvClosureNhd V W
    S : Subgroup G := { carrier := Set.iUnion fun n => HPow.hPow V (HAdd.hAdd n 1) …
    this : IsOpen (Set.iUnion fun n => HPow.hPow V (HAdd.hAdd n 1))
    ⊢ Exists fun H => HasSubset.Subset (↑H) W
  -/
  use ⟨S, this⟩
  have mulVpow (n : ℕ) : W * V ^ (n + 1) ⊆ W := by
    induction' n with n ih
    · simp [hV.mul]
    · rw [pow_succ, ← mul_assoc]
      exact (Set.mul_subset_mul_right ih).trans hV.mul
  have (n : ℕ) : V ^ (n + 1) ⊆ W * V ^ (n + 1) := by
    intro x xin
    rw [Set.mem_mul]
    use 1, einW, x, xin
    rw [one_mul]
  /-
    case h
    G : Type u_2
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    einW : Membership.mem W 1
    V : Set G
    hV : TopologicalGroup.mulInvClosureNhd V W
    S : Subgroup G := { carrier := Set.iUnion fun n => HPow.hPow V (HAdd.hAdd n 1) …
    this✝ : IsOpen (Set.iUnion fun n => HPow.hPow V (HAdd.hAdd n 1))
    mulVpow : ∀ (n : Nat), HasSubset.Subset (HMul.hMul W (HPow.hPow V (HAdd.hAdd n …
    this : ∀ (n : Nat), HasSubset.Subset (HPow.hPow V (HAdd.hAdd n 1)) (HMul.hMul  …
    ⊢ HasSubset.Subset (↑{ toSubgroup := S, isOpen' := this✝ }) W
  -/
  apply iUnion_subset fun i _ a ↦ mulVpow i (this i a)
  /-
    🎉 no goals
  -/


