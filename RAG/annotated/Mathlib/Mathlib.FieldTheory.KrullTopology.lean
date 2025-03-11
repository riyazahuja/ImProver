/-- Mapping intermediate fields along the identity does not change them -/
theorem IntermediateField.map_id {K L : Type*} [Field K] [Field L] [Algebra K L]
    (E : IntermediateField K L) : E.map (AlgHom.id K L) = E :=
  SetLike.coe_injective <| Set.image_id _


/-- Mapping a finite dimensional intermediate field along an algebra equivalence gives
a finite-dimensional intermediate field. -/
instance im_finiteDimensional {K L : Type*} [Field K] [Field L] [Algebra K L]
    {E : IntermediateField K L} (σ : L ≃ₐ[K] L) [FiniteDimensional K E] :
    FiniteDimensional K (E.map σ.toAlgHom) :=
  LinearEquiv.finiteDimensional (IntermediateField.intermediateFieldMap σ E).toLinearEquiv


/-- Given a field extension `L/K`, `finiteExts K L` is the set of
intermediate field extensions `L/E/K` such that `E/K` is finite -/
def finiteExts (K : Type*) [Field K] (L : Type*) [Field L] [Algebra K L] :
    Set (IntermediateField K L) :=
  {E | FiniteDimensional K E}


/-- Given a field extension `L/K`, `fixedByFinite K L` is the set of
subsets `Gal(L/E)` of `L ≃ₐ[K] L`, where `E/K` is finite -/
def fixedByFinite (K L : Type*) [Field K] [Field L] [Algebra K L] : Set (Subgroup (L ≃ₐ[K] L)) :=
  IntermediateField.fixingSubgroup '' finiteExts K L


/-- For a field extension `L/K`, the intermediate field `K` is finite-dimensional over `K` -/
theorem IntermediateField.finiteDimensional_bot (K L : Type*) [Field K] [Field L] [Algebra K L] :
    FiniteDimensional K (⊥ : IntermediateField K L) :=
  .of_rank_eq_one IntermediateField.rank_bot


/-- This lemma says that `Gal(L/K) = L ≃ₐ[K] L` -/
theorem IntermediateField.fixingSubgroup.bot {K L : Type*} [Field K] [Field L] [Algebra K L] :
    IntermediateField.fixingSubgroup (⊥ : IntermediateField K L) = ⊤ := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ⊢ Eq Bot.bot.fixingSubgroup Top.top
  -/
  ext f
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    f : AlgEquiv K L L
    ⊢ Iff (Membership.mem Bot.bot.fixingSubgroup f) (Membership.mem Top.top f)
  -/
  refine ⟨fun _ => Subgroup.mem_top _, fun _ => ?_⟩
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    f : AlgEquiv K L L
    x✝ : Membership.mem Top.top f
    ⊢ Membership.mem Bot.bot.fixingSubgroup f
  -/
  rintro ⟨x, hx : x ∈ (⊥ : IntermediateField K L)⟩
  /-
    case h.mk
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    f : AlgEquiv K L L
    x✝ : Membership.mem Top.top f
    x : L
    hx : Membership.mem Bot.bot x
    ⊢ Eq (HSMul.hSMul f ↑⟨x, hx⟩) ↑⟨x, hx⟩
  -/
  rw [IntermediateField.mem_bot] at hx
  /-
    case h.mk
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    f : AlgEquiv K L L
    x✝ : Membership.mem Top.top f
    x : L
    hx✝ : Membership.mem Bot.bot x
    hx : Membership.mem (Set.range ⇑(algebraMap K L)) x
    ⊢ Eq (HSMul.hSMul f ↑⟨x, hx✝⟩) ↑⟨x, hx✝⟩
  -/
  rcases hx with ⟨y, rfl⟩
  /-
    case h.mk.intro
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    f : AlgEquiv K L L
    x✝ : Membership.mem Top.top f
    y : K
    hx : Membership.mem Bot.bot ((algebraMap K L) y)
    ⊢ Eq (HSMul.hSMul f ↑⟨(algebraMap K L) y, hx⟩) ↑⟨(algebraMap K L) y, hx⟩
  -/
  exact f.commutes y
  /-
    🎉 no goals
  -/


/-- If `L/K` is a field extension, then we have `Gal(L/K) ∈ fixedByFinite K L` -/
theorem top_fixedByFinite {K L : Type*} [Field K] [Field L] [Algebra K L] :
    ⊤ ∈ fixedByFinite K L :=
  ⟨⊥, IntermediateField.finiteDimensional_bot K L, IntermediateField.fixingSubgroup.bot⟩


/-- If `E1` and `E2` are finite-dimensional intermediate fields, then so is their compositum.
This rephrases a result already in mathlib so that it is compatible with our type classes -/
theorem finiteDimensional_sup {K L : Type*} [Field K] [Field L] [Algebra K L]
    (E1 E2 : IntermediateField K L) (_ : FiniteDimensional K E1) (_ : FiniteDimensional K E2) :
    FiniteDimensional K (↥(E1 ⊔ E2)) :=
  IntermediateField.finiteDimensional_sup E1 E2


/-- An element of `L ≃ₐ[K] L` is in `Gal(L/E)` if and only if it fixes every element of `E`-/
theorem IntermediateField.mem_fixingSubgroup_iff {K L : Type*} [Field K] [Field L] [Algebra K L]
    (E : IntermediateField K L) (σ : L ≃ₐ[K] L) : σ ∈ E.fixingSubgroup ↔ ∀ x : L, x ∈ E → σ x = x :=
  ⟨fun hσ x hx => hσ ⟨x, hx⟩, fun h ⟨x, hx⟩ => h x hx⟩


/-- The map `E ↦ Gal(L/E)` is inclusion-reversing -/
theorem IntermediateField.fixingSubgroup.antimono {K L : Type*} [Field K] [Field L] [Algebra K L]
    {E1 E2 : IntermediateField K L} (h12 : E1 ≤ E2) : E2.fixingSubgroup ≤ E1.fixingSubgroup := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    E1 E2 : IntermediateField K L
    h12 : LE.le E1 E2
    ⊢ LE.le E2.fixingSubgroup E1.fixingSubgroup
  -/
  rintro σ hσ ⟨x, hx⟩
  /-
    case mk
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    E1 E2 : IntermediateField K L
    h12 : LE.le E1 E2
    σ : AlgEquiv K L L
    hσ : Membership.mem E2.fixingSubgroup σ
    x : L
    hx : Membership.mem (↑E1) x
    ⊢ Eq (HSMul.hSMul σ ↑⟨x, hx⟩) ↑⟨x, hx⟩
  -/
  exact hσ ⟨x, h12 hx⟩
  /-
    🎉 no goals
  -/


/-- Given a field extension `L/K`, `galBasis K L` is the filter basis on `L ≃ₐ[K] L` whose sets
are `Gal(L/E)` for intermediate fields `E` with `E/K` finite dimensional -/
def galBasis (K L : Type*) [Field K] [Field L] [Algebra K L] : FilterBasis (L ≃ₐ[K] L) where
  sets := (fun g => g.carrier) '' fixedByFinite K L
  nonempty := ⟨⊤, ⊤, top_fixedByFinite, rfl⟩
  inter_sets := by
    /-
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      ⊢ ∀ {x y : Set (AlgEquiv K L L)}, Membership.mem (Set.image (fun g => g.carrie …
    -/
    rintro X Y ⟨H1, ⟨E1, h_E1, rfl⟩, rfl⟩ ⟨H2, ⟨E2, h_E2, rfl⟩, rfl⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      E1 : IntermediateField K L
      h_E1 : Membership.mem (finiteExts K L) E1
      E2 : IntermediateField K L
      h_E2 : Membership.mem (finiteExts K L) E2
      ⊢ Exists fun z => And (Membership.mem (Set.image (fun g => g.carrier) (fixedBy …
    -/
    use (IntermediateField.fixingSubgroup (E1 ⊔ E2)).carrier
    /-
      case h
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      E1 : IntermediateField K L
      h_E1 : Membership.mem (finiteExts K L) E1
      E2 : IntermediateField K L
      h_E2 : Membership.mem (finiteExts K L) E2
      ⊢ And (Membership.mem (Set.image (fun g => g.carrier) (fixedByFinite K L)) (Ma …
    -/
    refine ⟨⟨_, ⟨_, finiteDimensional_sup E1 E2 h_E1 h_E2, rfl⟩, rfl⟩, ?_⟩
    /-
      case h
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      E1 : IntermediateField K L
      h_E1 : Membership.mem (finiteExts K L) E1
      E2 : IntermediateField K L
      h_E2 : Membership.mem (finiteExts K L) E2
      ⊢ HasSubset.Subset (Max.max E1 E2).fixingSubgroup.carrier (Inter.inter ((fun g …
    -/
    rw [Set.subset_inter_iff]
    exact
      ⟨IntermediateField.fixingSubgroup.antimono le_sup_left,
        IntermediateField.fixingSubgroup.antimono le_sup_right⟩


/-- A subset of `L ≃ₐ[K] L` is a member of `galBasis K L` if and only if it is the underlying set
of `Gal(L/E)` for some finite subextension `E/K`-/
theorem mem_galBasis_iff (K L : Type*) [Field K] [Field L] [Algebra K L] (U : Set (L ≃ₐ[K] L)) :
    U ∈ galBasis K L ↔ U ∈ (fun g => g.carrier) '' fixedByFinite K L :=
  Iff.rfl


/-- For a field extension `L/K`, `galGroupBasis K L` is the group filter basis on `L ≃ₐ[K] L`
whose sets are `Gal(L/E)` for finite subextensions `E/K` -/
def galGroupBasis (K L : Type*) [Field K] [Field L] [Algebra K L] :
    GroupFilterBasis (L ≃ₐ[K] L) where
  toFilterBasis := galBasis K L
  one' := fun ⟨H, _, h2⟩ => h2 ▸ H.one_mem
  mul' {U} hU :=
    ⟨U, hU, by
      /-
        K : Type u_1
        L : Type u_2
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        U : Set (AlgEquiv K L L)
        hU : Membership.mem (galBasis K L).sets U
        ⊢ HasSubset.Subset (HMul.hMul U U) U
      -/
      rcases hU with ⟨H, _, rfl⟩
      /-
        case intro.intro
        K : Type u_1
        L : Type u_2
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        H : Subgroup (AlgEquiv K L L)
        left✝ : Membership.mem (fixedByFinite K L) H
        ⊢ HasSubset.Subset (HMul.hMul ((fun g => g.carrier) H) ((fun g => g.carrier) H …
      -/
      rintro x ⟨a, haH, b, hbH, rfl⟩
      /-
        case intro.intro.intro.intro.intro.intro
        K : Type u_1
        L : Type u_2
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        H : Subgroup (AlgEquiv K L L)
        left✝ : Membership.mem (fixedByFinite K L) H
        a : AlgEquiv K L L
        haH : Membership.mem ((fun g => g.carrier) H) a
        b : AlgEquiv K L L
        hbH : Membership.mem ((fun g => g.carrier) H) b
        ⊢ Membership.mem ((fun g => g.carrier) H) ((fun x1 x2 => HMul.hMul x1 x2) a b)
      -/
      exact H.mul_mem haH hbH⟩
      /-
        🎉 no goals
      -/
  inv' {U} hU :=
    ⟨U, hU, by
      /-
        K : Type u_1
        L : Type u_2
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        U : Set (AlgEquiv K L L)
        hU : Membership.mem (galBasis K L).sets U
        ⊢ HasSubset.Subset U (Set.preimage (fun x => Inv.inv x) U)
      -/
      rcases hU with ⟨H, _, rfl⟩
      /-
        case intro.intro
        K : Type u_1
        L : Type u_2
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        H : Subgroup (AlgEquiv K L L)
        left✝ : Membership.mem (fixedByFinite K L) H
        ⊢ HasSubset.Subset ((fun g => g.carrier) H) (Set.preimage (fun x => Inv.inv x) …
      -/
      exact fun _ => H.inv_mem'⟩
      /-
        🎉 no goals
      -/
  conj' := by
    /-
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      ⊢ ∀ (x₀ : AlgEquiv K L L) {U : Set (AlgEquiv K L L)}, Membership.mem (galBasis …
    -/
    rintro σ U ⟨H, ⟨E, hE, rfl⟩, rfl⟩
    /-
      case intro.intro.intro.intro
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      E : IntermediateField K L
      hE : Membership.mem (finiteExts K L) E
      ⊢ Exists fun V => And (Membership.mem (galBasis K L).sets V) (HasSubset.Subset …
    -/
    let F : IntermediateField K L := E.map σ.symm.toAlgHom
    /-
      case intro.intro.intro.intro
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      E : IntermediateField K L
      hE : Membership.mem (finiteExts K L) E
      F : IntermediateField K L := IntermediateField.map (↑σ.symm) E
      ⊢ Exists fun V => And (Membership.mem (galBasis K L).sets V) (HasSubset.Subset …
    -/
    refine ⟨F.fixingSubgroup.carrier, ⟨⟨F.fixingSubgroup, ⟨F, ?_, rfl⟩, rfl⟩, fun g hg => ?_⟩⟩
      /-
        case intro.intro.intro.intro.refine_1
        K : Type u_1
        L : Type u_2
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        σ : AlgEquiv K L L
        E : IntermediateField K L
        hE : Membership.mem (finiteExts K L) E
        F : IntermediateField K L := IntermediateField.map (↑σ.symm) E
        ⊢ Membership.mem (finiteExts K L) F
      -/
    · have : FiniteDimensional K E := hE
      /-
        case intro.intro.intro.intro.refine_1
        K : Type u_1
        L : Type u_2
        inst✝² : Field K
        inst✝¹ : Field L
        inst✝ : Algebra K L
        σ : AlgEquiv K L L
        E : IntermediateField K L
        hE : Membership.mem (finiteExts K L) E
        F : IntermediateField K L := IntermediateField.map (↑σ.symm) E
        this : FiniteDimensional K (Subtype fun x => Membership.mem E x)
        ⊢ Membership.mem (finiteExts K L) F
      -/
      apply im_finiteDimensional σ.symm
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      E : IntermediateField K L
      hE : Membership.mem (finiteExts K L) E
      F : IntermediateField K L := IntermediateField.map (↑σ.symm) E
      g : AlgEquiv K L L
      hg : Membership.mem F.fixingSubgroup.carrier g
      ⊢ Membership.mem (Set.preimage (fun x => HMul.hMul (HMul.hMul σ x) (Inv.inv σ) …
    -/
    change σ * g * σ⁻¹ ∈ E.fixingSubgroup
    /-
      case intro.intro.intro.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      E : IntermediateField K L
      hE : Membership.mem (finiteExts K L) E
      F : IntermediateField K L := IntermediateField.map (↑σ.symm) E
      g : AlgEquiv K L L
      hg : Membership.mem F.fixingSubgroup.carrier g
      ⊢ Membership.mem E.fixingSubgroup (HMul.hMul (HMul.hMul σ g) (Inv.inv σ))
    -/
    rw [IntermediateField.mem_fixingSubgroup_iff]
    /-
      case intro.intro.intro.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      E : IntermediateField K L
      hE : Membership.mem (finiteExts K L) E
      F : IntermediateField K L := IntermediateField.map (↑σ.symm) E
      g : AlgEquiv K L L
      hg : Membership.mem F.fixingSubgroup.carrier g
      ⊢ ∀ (x : L), Membership.mem E x → Eq ((HMul.hMul (HMul.hMul σ g) (Inv.inv σ))  …
    -/
    intro x hx
    /-
      case intro.intro.intro.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      E : IntermediateField K L
      hE : Membership.mem (finiteExts K L) E
      F : IntermediateField K L := IntermediateField.map (↑σ.symm) E
      g : AlgEquiv K L L
      hg : Membership.mem F.fixingSubgroup.carrier g
      x : L
      hx : Membership.mem E x
      ⊢ Eq ((HMul.hMul (HMul.hMul σ g) (Inv.inv σ)) x) x
    -/
    change σ (g (σ⁻¹ x)) = x
    /-
      case intro.intro.intro.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      E : IntermediateField K L
      hE : Membership.mem (finiteExts K L) E
      F : IntermediateField K L := IntermediateField.map (↑σ.symm) E
      g : AlgEquiv K L L
      hg : Membership.mem F.fixingSubgroup.carrier g
      x : L
      hx : Membership.mem E x
      ⊢ Eq (σ (g ((Inv.inv σ) x))) x
    -/
    have h_in_F : σ⁻¹ x ∈ F := ⟨x, hx, by dsimp; rw [← AlgEquiv.invFun_eq_symm]; rfl⟩
    have h_g_fix : g (σ⁻¹ x) = σ⁻¹ x := by
      rw [Subgroup.mem_carrier, IntermediateField.mem_fixingSubgroup_iff F g] at hg
      exact hg (σ⁻¹ x) h_in_F
    /-
      case intro.intro.intro.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      E : IntermediateField K L
      hE : Membership.mem (finiteExts K L) E
      F : IntermediateField K L := IntermediateField.map (↑σ.symm) E
      g : AlgEquiv K L L
      hg : Membership.mem F.fixingSubgroup.carrier g
      x : L
      hx : Membership.mem E x
      h_in_F : Membership.mem F ((Inv.inv σ) x)
      h_g_fix : Eq (g ((Inv.inv σ) x)) ((Inv.inv σ) x)
      ⊢ Eq (σ (g ((Inv.inv σ) x))) x
    -/
    rw [h_g_fix]
    /-
      case intro.intro.intro.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      E : IntermediateField K L
      hE : Membership.mem (finiteExts K L) E
      F : IntermediateField K L := IntermediateField.map (↑σ.symm) E
      g : AlgEquiv K L L
      hg : Membership.mem F.fixingSubgroup.carrier g
      x : L
      hx : Membership.mem E x
      h_in_F : Membership.mem F ((Inv.inv σ) x)
      h_g_fix : Eq (g ((Inv.inv σ) x)) ((Inv.inv σ) x)
      ⊢ Eq (σ ((Inv.inv σ) x)) x
    -/
    change σ (σ⁻¹ x) = x
    /-
      case intro.intro.intro.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      E : IntermediateField K L
      hE : Membership.mem (finiteExts K L) E
      F : IntermediateField K L := IntermediateField.map (↑σ.symm) E
      g : AlgEquiv K L L
      hg : Membership.mem F.fixingSubgroup.carrier g
      x : L
      hx : Membership.mem E x
      h_in_F : Membership.mem F ((Inv.inv σ) x)
      h_g_fix : Eq (g ((Inv.inv σ) x)) ((Inv.inv σ) x)
      ⊢ Eq (σ ((Inv.inv σ) x)) x
    -/
    exact AlgEquiv.apply_symm_apply σ x
    /-
      🎉 no goals
    -/


/-- For a field extension `L/K`, `krullTopology K L` is the topological space structure on
`L ≃ₐ[K] L` induced by the group filter basis `galGroupBasis K L` -/
instance krullTopology (K L : Type*) [Field K] [Field L] [Algebra K L] :
    TopologicalSpace (L ≃ₐ[K] L) :=
  GroupFilterBasis.topology (galGroupBasis K L)


/-- For a field extension `L/K`, the Krull topology on `L ≃ₐ[K] L` makes it a topological group. -/
instance (K L : Type*) [Field K] [Field L] [Algebra K L] : TopologicalGroup (L ≃ₐ[K] L) :=
  GroupFilterBasis.isTopologicalGroup (galGroupBasis K L)


open scoped Topology in
lemma krullTopology_mem_nhds_one (K L : Type*) [Field K] [Field L] [Algebra K L]
    (s : Set (L ≃ₐ[K] L)) : s ∈ 𝓝 1 ↔ ∃ E : IntermediateField K L,
    FiniteDimensional K E ∧ (E.fixingSubgroup : Set (L ≃ₐ[K] L)) ⊆ s := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    s : Set (AlgEquiv K L L)
    ⊢ Iff (Membership.mem (nhds 1) s) (Exists fun E => And (FiniteDimensional K (S …
  -/
  rw [GroupFilterBasis.nhds_one_eq]
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    s : Set (AlgEquiv K L L)
    ⊢ Iff (Membership.mem GroupFilterBasis.toFilterBasis.filter s) (Exists fun E = …
  -/
  constructor
    /-
      case mp
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      s : Set (AlgEquiv K L L)
      ⊢ Membership.mem GroupFilterBasis.toFilterBasis.filter s → Exists fun E => And …
    -/
  · rintro ⟨-, ⟨-, ⟨E, fin, rfl⟩, rfl⟩, hE⟩
    /-
      case mp.intro.intro.intro.intro.intro.intro
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      s : Set (AlgEquiv K L L)
      E : IntermediateField K L
      fin : Membership.mem (finiteExts K L) E
      hE : HasSubset.Subset ((fun g => g.carrier) E.fixingSubgroup) s
      ⊢ Exists fun E => And (FiniteDimensional K (Subtype fun x => Membership.mem E  …
    -/
    exact ⟨E, fin, hE⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      s : Set (AlgEquiv K L L)
      ⊢ (Exists fun E => And (FiniteDimensional K (Subtype fun x => Membership.mem E …
    -/
  · rintro ⟨E, fin, hE⟩
    /-
      case mpr.intro.intro
      K : Type u_1
      L : Type u_2
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      s : Set (AlgEquiv K L L)
      E : IntermediateField K L
      fin : FiniteDimensional K (Subtype fun x => Membership.mem E x)
      hE : HasSubset.Subset (↑E.fixingSubgroup) s
      ⊢ Membership.mem GroupFilterBasis.toFilterBasis.filter s
    -/
    exact ⟨E.fixingSubgroup, ⟨E.fixingSubgroup, ⟨E, fin, rfl⟩, rfl⟩, hE⟩
    /-
      🎉 no goals
    -/


/-- Let `L/E/K` be a tower of fields with `E/K` finite. Then `Gal(L/E)` is an open subgroup of
  `L ≃ₐ[K] L`. -/
theorem IntermediateField.fixingSubgroup_isOpen {K L : Type*} [Field K] [Field L] [Algebra K L]
    (E : IntermediateField K L) [FiniteDimensional K E] :
    IsOpen (E.fixingSubgroup : Set (L ≃ₐ[K] L)) := by
  have h_basis : E.fixingSubgroup.carrier ∈ galGroupBasis K L :=
    ⟨E.fixingSubgroup, ⟨E, ‹_›, rfl⟩, rfl⟩
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    E : IntermediateField K L
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem E x)
    h_basis : Membership.mem (galGroupBasis K L) E.fixingSubgroup.carrier
    ⊢ IsOpen ↑E.fixingSubgroup
  -/
  have h_nhd := GroupFilterBasis.mem_nhds_one (galGroupBasis K L) h_basis
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    E : IntermediateField K L
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem E x)
    h_basis : Membership.mem (galGroupBasis K L) E.fixingSubgroup.carrier
    h_nhd : Membership.mem (nhds 1) E.fixingSubgroup.carrier
    ⊢ IsOpen ↑E.fixingSubgroup
  -/
  exact Subgroup.isOpen_of_mem_nhds _ h_nhd
  /-
    🎉 no goals
  -/


/-- Given a tower of fields `L/E/K`, with `E/K` finite, the subgroup `Gal(L/E) ≤ L ≃ₐ[K] L` is
  closed. -/
theorem IntermediateField.fixingSubgroup_isClosed {K L : Type*} [Field K] [Field L] [Algebra K L]
    (E : IntermediateField K L) [FiniteDimensional K E] :
    IsClosed (E.fixingSubgroup : Set (L ≃ₐ[K] L)) :=
  OpenSubgroup.isClosed ⟨E.fixingSubgroup, E.fixingSubgroup_isOpen⟩


/-- If `L/K` is an algebraic extension, then the Krull topology on `L ≃ₐ[K] L` is Hausdorff. -/
theorem krullTopology_t2 {K L : Type*} [Field K] [Field L] [Algebra K L]
    [Algebra.IsIntegral K L] : T2Space (L ≃ₐ[K] L) :=
  { t2 := fun f g hfg => by
      /-
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        ⊢ (fun x y => Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) ( …
      -/
      let φ := f⁻¹ * g
      /-
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        ⊢ (fun x y => Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) ( …
      -/
      cases' DFunLike.exists_ne hfg with x hx
      have hφx : φ x ≠ x := by
        apply ne_of_apply_ne f
        change f (f.symm (g x)) ≠ f x
        rw [AlgEquiv.apply_symm_apply f (g x), ne_comm]
        exact hx
      /-
        case intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        ⊢ (fun x y => Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) ( …
      -/
      let E : IntermediateField K L := IntermediateField.adjoin K {x}
      let h_findim : FiniteDimensional K E := IntermediateField.adjoin.finiteDimensional
        (Algebra.IsIntegral.isIntegral x)
      /-
        case intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        ⊢ (fun x y => Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) ( …
      -/
      let H := E.fixingSubgroup
      /-
        case intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        ⊢ (fun x y => Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) ( …
      -/
      have h_basis : (H : Set (L ≃ₐ[K] L)) ∈ galGroupBasis K L := ⟨H, ⟨E, ⟨h_findim, rfl⟩⟩, rfl⟩
      /-
        case intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        ⊢ (fun x y => Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) ( …
      -/
      have h_nhd := GroupFilterBasis.mem_nhds_one (galGroupBasis K L) h_basis
      /-
        case intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        h_nhd : Membership.mem (nhds 1) ↑H
        ⊢ (fun x y => Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) ( …
      -/
      rw [mem_nhds_iff] at h_nhd
      /-
        case intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        h_nhd : Exists fun t => And (HasSubset.Subset t ↑H) (And (IsOpen t) (Membershi …
        ⊢ (fun x y => Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) ( …
      -/
      rcases h_nhd with ⟨W, hWH, hW_open, hW_1⟩
      refine ⟨f • W, g • W,
        ⟨hW_open.leftCoset f, hW_open.leftCoset g, ⟨1, hW_1, mul_one _⟩, ⟨1, hW_1, mul_one _⟩, ?_⟩⟩
      /-
        case intro.intro.intro.intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        W : Set (AlgEquiv K L L)
        hWH : HasSubset.Subset W ↑H
        hW_open : IsOpen W
        hW_1 : Membership.mem W 1
        ⊢ Disjoint (HSMul.hSMul f W) (HSMul.hSMul g W)
      -/
      rw [Set.disjoint_left]
      /-
        case intro.intro.intro.intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        W : Set (AlgEquiv K L L)
        hWH : HasSubset.Subset W ↑H
        hW_open : IsOpen W
        hW_1 : Membership.mem W 1
        ⊢ ∀ ⦃a : AlgEquiv K L L⦄, Membership.mem (HSMul.hSMul f W) a → Not (Membership …
      -/
      rintro σ ⟨w1, hw1, h⟩ ⟨w2, hw2, rfl⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        W : Set (AlgEquiv K L L)
        hWH : HasSubset.Subset W ↑H
        hW_open : IsOpen W
        hW_1 : Membership.mem W 1
        w1 : AlgEquiv K L L
        hw1 : Membership.mem W w1
        w2 : AlgEquiv K L L
        hw2 : Membership.mem W w2
        h : Eq ((fun x => HSMul.hSMul f x) w1) ((fun x => HSMul.hSMul g x) w2)
        ⊢ False
      -/
      dsimp at h
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        W : Set (AlgEquiv K L L)
        hWH : HasSubset.Subset W ↑H
        hW_open : IsOpen W
        hW_1 : Membership.mem W 1
        w1 : AlgEquiv K L L
        hw1 : Membership.mem W w1
        w2 : AlgEquiv K L L
        hw2 : Membership.mem W w2
        h : Eq (HMul.hMul f w1) (HMul.hMul g w2)
        ⊢ False
      -/
      rw [eq_inv_mul_iff_mul_eq.symm, ← mul_assoc, mul_inv_eq_iff_eq_mul.symm] at h
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        W : Set (AlgEquiv K L L)
        hWH : HasSubset.Subset W ↑H
        hW_open : IsOpen W
        hW_1 : Membership.mem W 1
        w1 : AlgEquiv K L L
        hw1 : Membership.mem W w1
        w2 : AlgEquiv K L L
        hw2 : Membership.mem W w2
        h : Eq (HMul.hMul w1 (Inv.inv w2)) (HMul.hMul (Inv.inv f) g)
        ⊢ False
      -/
      have h_in_H : w1 * w2⁻¹ ∈ H := H.mul_mem (hWH hw1) (H.inv_mem (hWH hw2))
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        W : Set (AlgEquiv K L L)
        hWH : HasSubset.Subset W ↑H
        hW_open : IsOpen W
        hW_1 : Membership.mem W 1
        w1 : AlgEquiv K L L
        hw1 : Membership.mem W w1
        w2 : AlgEquiv K L L
        hw2 : Membership.mem W w2
        h : Eq (HMul.hMul w1 (Inv.inv w2)) (HMul.hMul (Inv.inv f) g)
        h_in_H : Membership.mem H (HMul.hMul w1 (Inv.inv w2))
        ⊢ False
      -/
      rw [h] at h_in_H
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        W : Set (AlgEquiv K L L)
        hWH : HasSubset.Subset W ↑H
        hW_open : IsOpen W
        hW_1 : Membership.mem W 1
        w1 : AlgEquiv K L L
        hw1 : Membership.mem W w1
        w2 : AlgEquiv K L L
        hw2 : Membership.mem W w2
        h : Eq (HMul.hMul w1 (Inv.inv w2)) (HMul.hMul (Inv.inv f) g)
        h_in_H : Membership.mem H (HMul.hMul (Inv.inv f) g)
        ⊢ False
      -/
      change φ ∈ E.fixingSubgroup at h_in_H
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        W : Set (AlgEquiv K L L)
        hWH : HasSubset.Subset W ↑H
        hW_open : IsOpen W
        hW_1 : Membership.mem W 1
        w1 : AlgEquiv K L L
        hw1 : Membership.mem W w1
        w2 : AlgEquiv K L L
        hw2 : Membership.mem W w2
        h : Eq (HMul.hMul w1 (Inv.inv w2)) (HMul.hMul (Inv.inv f) g)
        h_in_H : Membership.mem E.fixingSubgroup φ
        ⊢ False
      -/
      rw [IntermediateField.mem_fixingSubgroup_iff] at h_in_H
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        W : Set (AlgEquiv K L L)
        hWH : HasSubset.Subset W ↑H
        hW_open : IsOpen W
        hW_1 : Membership.mem W 1
        w1 : AlgEquiv K L L
        hw1 : Membership.mem W w1
        w2 : AlgEquiv K L L
        hw2 : Membership.mem W w2
        h : Eq (HMul.hMul w1 (Inv.inv w2)) (HMul.hMul (Inv.inv f) g)
        h_in_H : ∀ (x : L), Membership.mem E x → Eq (φ x) x
        ⊢ False
      -/
      specialize h_in_H x
      have hxE : x ∈ E := by
        apply IntermediateField.subset_adjoin
        apply Set.mem_singleton
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        K : Type u_1
        L : Type u_2
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : Algebra.IsIntegral K L
        f g : AlgEquiv K L L
        hfg : Ne f g
        φ : AlgEquiv K L L := HMul.hMul (Inv.inv f) g
        x : L
        hx : Ne (f x) (g x)
        hφx : Ne (φ x) x
        E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
        h_findim : FiniteDimensional K (Subtype fun x => Membership.mem E x) := Interm …
        H : Subgroup (AlgEquiv K L L) := E.fixingSubgroup
        h_basis : Membership.mem (galGroupBasis K L) ↑H
        W : Set (AlgEquiv K L L)
        hWH : HasSubset.Subset W ↑H
        hW_open : IsOpen W
        hW_1 : Membership.mem W 1
        w1 : AlgEquiv K L L
        hw1 : Membership.mem W w1
        w2 : AlgEquiv K L L
        hw2 : Membership.mem W w2
        h : Eq (HMul.hMul w1 (Inv.inv w2)) (HMul.hMul (Inv.inv f) g)
        h_in_H : Membership.mem E x → Eq (φ x) x
        hxE : Membership.mem E x
        ⊢ False
      -/
      exact hφx (h_in_H hxE) }
      /-
        🎉 no goals
      -/


/-- If `L/K` is an algebraic field extension, then the Krull topology on `L ≃ₐ[K] L` is
  totally disconnected. -/
theorem krullTopology_totallyDisconnected {K L : Type*} [Field K] [Field L] [Algebra K L]
    [Algebra.IsIntegral K L] : IsTotallyDisconnected (Set.univ : Set (L ≃ₐ[K] L)) := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Algebra.IsIntegral K L
    ⊢ IsTotallyDisconnected Set.univ
  -/
  apply isTotallyDisconnected_of_isClopen_set
  /-
    case hX
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Algebra.IsIntegral K L
    ⊢ Pairwise fun x y => Exists fun U => And (IsClopen U) (And (Membership.mem U  …
  -/
  intro σ τ h_diff
  /-
    case hX
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Algebra.IsIntegral K L
    σ τ : AlgEquiv K L L
    h_diff : Ne σ τ
    ⊢ Exists fun U => And (IsClopen U) (And (Membership.mem U σ) (Not (Membership. …
  -/
  have hστ : σ⁻¹ * τ ≠ 1 := by rwa [Ne, inv_mul_eq_one]
  /-
    case hX
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Algebra.IsIntegral K L
    σ τ : AlgEquiv K L L
    h_diff : Ne σ τ
    hστ : Ne (HMul.hMul (Inv.inv σ) τ) 1
    ⊢ Exists fun U => And (IsClopen U) (And (Membership.mem U σ) (Not (Membership. …
  -/
  rcases DFunLike.exists_ne hστ with ⟨x, hx : (σ⁻¹ * τ) x ≠ x⟩
  /-
    case hX.intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Algebra.IsIntegral K L
    σ τ : AlgEquiv K L L
    h_diff : Ne σ τ
    hστ : Ne (HMul.hMul (Inv.inv σ) τ) 1
    x : L
    hx : Ne ((HMul.hMul (Inv.inv σ) τ) x) x
    ⊢ Exists fun U => And (IsClopen U) (And (Membership.mem U σ) (Not (Membership. …
  -/
  let E := IntermediateField.adjoin K ({x} : Set L)
  haveI := IntermediateField.adjoin.finiteDimensional
    (Algebra.IsIntegral.isIntegral (R := K) x)
  refine ⟨σ • E.fixingSubgroup,
    ⟨E.fixingSubgroup_isClosed.leftCoset σ, E.fixingSubgroup_isOpen.leftCoset σ⟩,
    ⟨1, E.fixingSubgroup.one_mem', mul_one σ⟩, ?_⟩
  simp only [mem_leftCoset_iff, SetLike.mem_coe, IntermediateField.mem_fixingSubgroup_iff,
    not_forall]
  /-
    case hX.intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Algebra.IsIntegral K L
    σ τ : AlgEquiv K L L
    h_diff : Ne σ τ
    hστ : Ne (HMul.hMul (Inv.inv σ) τ) 1
    x : L
    hx : Ne ((HMul.hMul (Inv.inv σ) τ) x) x
    E : IntermediateField K L := IntermediateField.adjoin K (Singleton.singleton x)
    this : FiniteDimensional K (Subtype fun x_1 => Membership.mem (IntermediateFie …
    ⊢ Exists fun x => Exists fun x_1 => Not (Eq ((HMul.hMul (Inv.inv σ) τ) x) x)
  -/
  exact ⟨x, IntermediateField.mem_adjoin_simple_self K x, hx⟩
  /-
    🎉 no goals
  -/


@[simp] lemma IntermediateField.fixingSubgroup_top (K L : Type*) [Field K] [Field L] [Algebra K L] :
    IntermediateField.fixingSubgroup (⊤ : IntermediateField K L) = ⊥ := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ⊢ Eq Top.top.fixingSubgroup Bot.bot
  -/
  ext
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x✝ : AlgEquiv K L L
    ⊢ Iff (Membership.mem Top.top.fixingSubgroup x✝) (Membership.mem Bot.bot x✝)
  -/
  simp [mem_fixingSubgroup_iff, DFunLike.ext_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma IntermediateField.fixingSubgroup_bot (K L : Type*) [Field K] [Field L] [Algebra K L] :
    IntermediateField.fixingSubgroup (⊥ : IntermediateField K L) = ⊤ := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ⊢ Eq Bot.bot.fixingSubgroup Top.top
  -/
  ext
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x✝ : AlgEquiv K L L
    ⊢ Iff (Membership.mem Bot.bot.fixingSubgroup x✝) (Membership.mem Top.top x✝)
  -/
  simp [mem_fixingSubgroup_iff, mem_bot]
  /-
    🎉 no goals
  -/


instance krullTopology_discreteTopology_of_finiteDimensional (K L : Type) [Field K] [Field L]
    [Algebra K L] [FiniteDimensional K L] : DiscreteTopology (L ≃ₐ[K] L) := by
  /-
    K L : Type
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    ⊢ DiscreteTopology (AlgEquiv K L L)
  -/
  rw [discreteTopology_iff_isOpen_singleton_one]
  /-
    K L : Type
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    ⊢ IsOpen (Singleton.singleton 1)
  -/
  change IsOpen ((⊥ : Subgroup (L ≃ₐ[K] L)) : Set (L ≃ₐ[K] L))
  /-
    K L : Type
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    ⊢ IsOpen ↑Bot.bot
  -/
  rw [← IntermediateField.fixingSubgroup_top]
  /-
    K L : Type
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    ⊢ IsOpen ↑Top.top.fixingSubgroup
  -/
  exact IntermediateField.fixingSubgroup_isOpen ⊤
  /-
    🎉 no goals
  -/

