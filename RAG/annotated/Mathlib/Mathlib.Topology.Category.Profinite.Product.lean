/-- The object part of the functor `indexFunctor : (Finset ι)ᵒᵖ ⥤ Profinite`. -/
def obj : Set ((i : {i : ι // J i}) → X i) := ContinuousMap.precomp (Subtype.val (p := J)) '' C


/-- The projection maps in the limit cone `indexCone`. -/
def π_app : C(C, obj C J) :=
  ⟨Set.MapsTo.restrict (precomp (Subtype.val (p := J))) _ _ (Set.mapsTo_image _ _),
    Continuous.restrict _ (Pi.continuous_precomp' _)⟩


/-- The morphism part of the functor `indexFunctor : (Finset ι)ᵒᵖ ⥤ Profinite`. -/
def map (h : ∀ i, J i → K i) : C(obj C K, obj C J) :=
  ⟨Set.MapsTo.restrict (precomp (Set.inclusion h)) _ _ (fun _ hx ↦ by
    /-
      ι : Type u
      X : ι → Type
      inst✝ : (i : ι) → TopologicalSpace (X i)
      C : Set ((i : ι) → X i)
      J K : ι → Prop
      h : ∀ (i : ι), J i → K i
      x✝ : (i : ↑K) → X ↑i
      hx : Membership.mem (Profinite.IndexFunctor.obj C K) x✝
      ⊢ Membership.mem (Profinite.IndexFunctor.obj C J) ((ContinuousMap.precomp (Set …
    -/
    obtain ⟨y, hy⟩ := hx
    /-
      case intro
      ι : Type u
      X : ι → Type
      inst✝ : (i : ι) → TopologicalSpace (X i)
      C : Set ((i : ι) → X i)
      J K : ι → Prop
      h : ∀ (i : ι), J i → K i
      x✝ : (i : ↑K) → X ↑i
      y : (i : ι) → X i
      hy : And (Membership.mem C y) (Eq ((ContinuousMap.precomp Subtype.val) y) x✝)
      ⊢ Membership.mem (Profinite.IndexFunctor.obj C J) ((ContinuousMap.precomp (Set …
    -/
    rw [← hy.2]
    /-
      case intro
      ι : Type u
      X : ι → Type
      inst✝ : (i : ι) → TopologicalSpace (X i)
      C : Set ((i : ι) → X i)
      J K : ι → Prop
      h : ∀ (i : ι), J i → K i
      x✝ : (i : ↑K) → X ↑i
      y : (i : ι) → X i
      hy : And (Membership.mem C y) (Eq ((ContinuousMap.precomp Subtype.val) y) x✝)
      ⊢ Membership.mem (Profinite.IndexFunctor.obj C J) ((ContinuousMap.precomp (Set …
    -/
    exact ⟨y, hy.1, rfl⟩), Continuous.restrict _ (Pi.continuous_precomp' _)⟩
    /-
      🎉 no goals
    -/


theorem surjective_π_app :
    Function.Surjective (π_app C J) := by
  /-
    ι : Type u
    X : ι → Type
    inst✝ : (i : ι) → TopologicalSpace (X i)
    C : Set ((i : ι) → X i)
    J : ι → Prop
    ⊢ Function.Surjective ⇑(Profinite.IndexFunctor.π_app C J)
  -/
  intro x
  /-
    ι : Type u
    X : ι → Type
    inst✝ : (i : ι) → TopologicalSpace (X i)
    C : Set ((i : ι) → X i)
    J : ι → Prop
    x : ↑(Profinite.IndexFunctor.obj C J)
    ⊢ Exists fun a => Eq ((Profinite.IndexFunctor.π_app C J) a) x
  -/
  obtain ⟨y, hy⟩ := x.prop
  /-
    case intro
    ι : Type u
    X : ι → Type
    inst✝ : (i : ι) → TopologicalSpace (X i)
    C : Set ((i : ι) → X i)
    J : ι → Prop
    x : ↑(Profinite.IndexFunctor.obj C J)
    y : (i : ι) → X i
    hy : And (Membership.mem C y) (Eq ((ContinuousMap.precomp Subtype.val) y) ↑x)
    ⊢ Exists fun a => Eq ((Profinite.IndexFunctor.π_app C J) a) x
  -/
  exact ⟨⟨y, hy.1⟩, Subtype.ext hy.2⟩
  /-
    🎉 no goals
  -/


theorem map_comp_π_app (h : ∀ i, J i → K i) : map C h ∘ π_app C K = π_app C J := rfl


theorem eq_of_forall_π_app_eq (a b : C)
    (h : ∀ (J : Finset ι), π_app C (· ∈ J) a = π_app C (· ∈ J) b) : a = b := by
  /-
    ι : Type u
    X : ι → Type
    inst✝ : (i : ι) → TopologicalSpace (X i)
    C : Set ((i : ι) → X i)
    a b : ↑C
    h : ∀ (J : Finset ι), Eq ((Profinite.IndexFunctor.π_app C fun x => Membership. …
    ⊢ Eq a b
  -/
  ext i
  /-
    case a.h
    ι : Type u
    X : ι → Type
    inst✝ : (i : ι) → TopologicalSpace (X i)
    C : Set ((i : ι) → X i)
    a b : ↑C
    h : ∀ (J : Finset ι), Eq ((Profinite.IndexFunctor.π_app C fun x => Membership. …
    i : ι
    ⊢ Eq (↑a i) (↑b i)
  -/
  specialize h ({i} : Finset ι)
  /-
    case a.h
    ι : Type u
    X : ι → Type
    inst✝ : (i : ι) → TopologicalSpace (X i)
    C : Set ((i : ι) → X i)
    a b : ↑C
    i : ι
    h : Eq ((Profinite.IndexFunctor.π_app C fun x => Membership.mem (Singleton.sin …
    ⊢ Eq (↑a i) (↑b i)
  -/
  rw [Subtype.ext_iff] at h
  simp only [π_app, ContinuousMap.precomp, ContinuousMap.coe_mk,
    Set.MapsTo.val_restrict_apply] at h
  /-
    case a.h
    ι : Type u
    X : ι → Type
    inst✝ : (i : ι) → TopologicalSpace (X i)
    C : Set ((i : ι) → X i)
    a b : ↑C
    i : ι
    h : Eq (fun j => ↑a ↑j) fun j => ↑b ↑j
    ⊢ Eq (↑a i) (↑b i)
  -/
  exact congr_fun h ⟨i, Finset.mem_singleton.mpr rfl⟩
  /-
    🎉 no goals
  -/


/-- The functor from the poset of finsets of `ι` to  `Profinite`, indexing the limit. -/
noncomputable
def indexFunctor (hC : IsCompact C) : (Finset ι)ᵒᵖ ⥤ Profinite.{u} where
  obj J := @Profinite.of (obj C (· ∈ (unop J))) _
        /-
          ι : Type u
          X : ι → Type
          inst✝² : (i : ι) → TopologicalSpace (X i)
          C : Set ((i : ι) → X i)
          J✝ K : ι → Prop
          inst✝¹ : ∀ (i : ι), T2Space (X i)
          inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
          hC : IsCompact C
          J : Opposite (Finset ι)
          ⊢ CompactSpace ↑(Profinite.IndexFunctor.obj C fun x => Membership.mem (Opposit …
        -/
    (by rw [← isCompact_iff_compactSpace]; exact hC.image (Pi.continuous_precomp' _)) _ _
                                           /-
                                             🎉 no goals
                                           -/
  map h := map C (leOfHom h.unop)


/-- The limit cone on `indexFunctor` -/
noncomputable
def indexCone (hC : IsCompact C) : Cone (indexFunctor hC) where
                              /-
                                ι : Type u
                                X : ι → Type
                                inst✝² : (i : ι) → TopologicalSpace (X i)
                                C : Set ((i : ι) → X i)
                                J K : ι → Prop
                                inst✝¹ : ∀ (i : ι), T2Space (X i)
                                inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
                                hC : IsCompact C
                                ⊢ CompactSpace ↑C
                              -/
  pt := @Profinite.of C _ (by rwa [← isCompact_iff_compactSpace]) _ _
                              /-
                                🎉 no goals
                              -/
  π := { app := fun J ↦ π_app C (· ∈ unop J) }


instance isIso_indexCone_lift :
    IsIso ((limitConeIsLimit.{u, u} (indexFunctor hC)).lift (indexCone hC)) :=
                               /-
                                 ι : Type u
                                 X : ι → Type
                                 inst✝² : (i : ι) → TopologicalSpace (X i)
                                 C : Set ((i : ι) → X i)
                                 J K : ι → Prop
                                 inst✝¹ : ∀ (i : ι), T2Space (X i)
                                 inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
                                 hC : IsCompact C
                                 ⊢ CompactSpace ↑C
                               -/
  haveI : CompactSpace C := by rwa [← isCompact_iff_compactSpace]
                               /-
                                 🎉 no goals
                               -/
  CompHausLike.isIso_of_bijective _
    (by
      /-
        ι : Type u
        X : ι → Type
        inst✝² : (i : ι) → TopologicalSpace (X i)
        C : Set ((i : ι) → X i)
        J K : ι → Prop
        inst✝¹ : ∀ (i : ι), T2Space (X i)
        inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
        hC : IsCompact C
        this : CompactSpace ↑C
        ⊢ Function.Bijective ⇑((Profinite.limitConeIsLimit (Profinite.indexFunctor hC) …
      -/
      refine ⟨fun a b h ↦ ?_, fun a ↦ ?_⟩
        /-
          case refine_1
          ι : Type u
          X : ι → Type
          inst✝² : (i : ι) → TopologicalSpace (X i)
          C : Set ((i : ι) → X i)
          J K : ι → Prop
          inst✝¹ : ∀ (i : ι), T2Space (X i)
          inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
          hC : IsCompact C
          this : CompactSpace ↑C
          a b : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑ …
          h : Eq (((Profinite.limitConeIsLimit (Profinite.indexFunctor hC)).lift (Profin …
          ⊢ Eq a b
        -/
      · refine eq_of_forall_π_app_eq a b (fun J ↦ ?_)
        /-
          case refine_1
          ι : Type u
          X : ι → Type
          inst✝² : (i : ι) → TopologicalSpace (X i)
          C : Set ((i : ι) → X i)
          J✝ K : ι → Prop
          inst✝¹ : ∀ (i : ι), T2Space (X i)
          inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
          hC : IsCompact C
          this : CompactSpace ↑C
          a b : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑ …
          h : Eq (((Profinite.limitConeIsLimit (Profinite.indexFunctor hC)).lift (Profin …
          J : Finset ι
          ⊢ Eq ((Profinite.IndexFunctor.π_app C fun x => Membership.mem J x) a) ((Profin …
        -/
        apply_fun fun f : (limitCone.{u, u} (indexFunctor hC)).pt => f.val (op J) at h
        /-
          case refine_1
          ι : Type u
          X : ι → Type
          inst✝² : (i : ι) → TopologicalSpace (X i)
          C : Set ((i : ι) → X i)
          J✝ K : ι → Prop
          inst✝¹ : ∀ (i : ι), T2Space (X i)
          inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
          hC : IsCompact C
          this : CompactSpace ↑C
          a b : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑ …
          J : Finset ι
          h : Eq (↑(((Profinite.limitConeIsLimit (Profinite.indexFunctor hC)).lift (Prof …
          ⊢ Eq ((Profinite.IndexFunctor.π_app C fun x => Membership.mem J x) a) ((Profin …
        -/
        exact h
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          ι : Type u
          X : ι → Type
          inst✝² : (i : ι) → TopologicalSpace (X i)
          C : Set ((i : ι) → X i)
          J K : ι → Prop
          inst✝¹ : ∀ (i : ι), T2Space (X i)
          inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
          hC : IsCompact C
          this : CompactSpace ↑C
          a : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
          ⊢ Exists fun a_1 => Eq (((Profinite.limitConeIsLimit (Profinite.indexFunctor h …
        -/
      · rsuffices ⟨b, hb⟩ : ∃ (x : C), ∀ (J : Finset ι), π_app C (· ∈ J) x = a.val (op J)
          /-
            case refine_2.intro
            ι : Type u
            X : ι → Type
            inst✝² : (i : ι) → TopologicalSpace (X i)
            C : Set ((i : ι) → X i)
            J K : ι → Prop
            inst✝¹ : ∀ (i : ι), T2Space (X i)
            inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
            hC : IsCompact C
            this : CompactSpace ↑C
            a : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
            b : ↑C
            hb : ∀ (J : Finset ι), Eq ((Profinite.IndexFunctor.π_app C fun x => Membership …
            ⊢ Exists fun a_1 => Eq (((Profinite.limitConeIsLimit (Profinite.indexFunctor h …
          -/
        · use b
          /-
            case h
            ι : Type u
            X : ι → Type
            inst✝² : (i : ι) → TopologicalSpace (X i)
            C : Set ((i : ι) → X i)
            J K : ι → Prop
            inst✝¹ : ∀ (i : ι), T2Space (X i)
            inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
            hC : IsCompact C
            this : CompactSpace ↑C
            a : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
            b : ↑C
            hb : ∀ (J : Finset ι), Eq ((Profinite.IndexFunctor.π_app C fun x => Membership …
            ⊢ Eq (((Profinite.limitConeIsLimit (Profinite.indexFunctor hC)).lift (Profinit …
          -/
          apply Subtype.ext
          /-
            case h.a
            ι : Type u
            X : ι → Type
            inst✝² : (i : ι) → TopologicalSpace (X i)
            C : Set ((i : ι) → X i)
            J K : ι → Prop
            inst✝¹ : ∀ (i : ι), T2Space (X i)
            inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
            hC : IsCompact C
            this : CompactSpace ↑C
            a : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
            b : ↑C
            hb : ∀ (J : Finset ι), Eq ((Profinite.IndexFunctor.π_app C fun x => Membership …
            ⊢ Eq ↑(((Profinite.limitConeIsLimit (Profinite.indexFunctor hC)).lift (Profini …
          -/
          apply funext
          /-
            case h.a.h
            ι : Type u
            X : ι → Type
            inst✝² : (i : ι) → TopologicalSpace (X i)
            C : Set ((i : ι) → X i)
            J K : ι → Prop
            inst✝¹ : ∀ (i : ι), T2Space (X i)
            inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
            hC : IsCompact C
            this : CompactSpace ↑C
            a : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
            b : ↑C
            hb : ∀ (J : Finset ι), Eq ((Profinite.IndexFunctor.π_app C fun x => Membership …
            ⊢ ∀ (x : Opposite (Finset ι)), Eq (↑(((Profinite.limitConeIsLimit (Profinite.i …
          -/
          intro J
          /-
            case h.a.h
            ι : Type u
            X : ι → Type
            inst✝² : (i : ι) → TopologicalSpace (X i)
            C : Set ((i : ι) → X i)
            J✝ K : ι → Prop
            inst✝¹ : ∀ (i : ι), T2Space (X i)
            inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
            hC : IsCompact C
            this : CompactSpace ↑C
            a : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
            b : ↑C
            hb : ∀ (J : Finset ι), Eq ((Profinite.IndexFunctor.π_app C fun x => Membership …
            J : Opposite (Finset ι)
            ⊢ Eq (↑(((Profinite.limitConeIsLimit (Profinite.indexFunctor hC)).lift (Profin …
          -/
          exact hb (unop J)
          /-
            🎉 no goals
          -/
        have hc : ∀ (J : Finset ι) s, IsClosed ((π_app C (· ∈ J)) ⁻¹' {s}) := by
          intro J s
          refine IsClosed.preimage (π_app C (· ∈ J)).continuous ?_
          exact T1Space.t1 s
        have H₁ : ∀ (Q₁ Q₂ : Finset ι), Q₁ ≤ Q₂ →
            π_app C (· ∈ Q₁) ⁻¹' {a.val (op Q₁)} ⊇
            π_app C (· ∈ Q₂) ⁻¹' {a.val (op Q₂)} := by
          intro J K h x hx
          simp only [Set.mem_preimage, Set.mem_singleton_iff] at hx ⊢
          rw [← map_comp_π_app C h, Function.comp_apply,
            hx, ← a.prop (homOfLE h).op]
          rfl
        obtain ⟨x, hx⟩ :
            Set.Nonempty (⋂ (J : Finset ι), π_app C (· ∈ J) ⁻¹' {a.val (op J)}) :=
          IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed
            (fun J : Finset ι => π_app C (· ∈ J) ⁻¹' {a.val (op J)}) (directed_of_isDirected_le H₁)
            (fun J => (Set.singleton_nonempty _).preimage (surjective_π_app _))
            (fun J => (hc J (a.val (op J))).isCompact) fun J => hc J (a.val (op J))
        /-
          case intro
          ι : Type u
          X : ι → Type
          inst✝² : (i : ι) → TopologicalSpace (X i)
          C : Set ((i : ι) → X i)
          J K : ι → Prop
          inst✝¹ : ∀ (i : ι), T2Space (X i)
          inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
          hC : IsCompact C
          this : CompactSpace ↑C
          a : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
          hc : ∀ (J : Finset ι) (s : ↑(Profinite.IndexFunctor.obj C fun x => Membership. …
          H₁ : ∀ (Q₁ Q₂ : Finset ι), LE.le Q₁ Q₂ → Superset (Set.preimage (⇑(Profinite.I …
          x : ↑C
          hx : Membership.mem (Set.iInter fun J => Set.preimage (⇑(Profinite.IndexFuncto …
          ⊢ Exists fun x => ∀ (J : Finset ι), Eq ((Profinite.IndexFunctor.π_app C fun x  …
        -/
        exact ⟨x, Set.mem_iInter.1 hx⟩)
        /-
          🎉 no goals
        -/


/-- The canonical map from `C` to the explicit limit as an isomorphism. -/
noncomputable
def isoindexConeLift :
                          /-
                            ι : Type u
                            X : ι → Type
                            inst✝² : (i : ι) → TopologicalSpace (X i)
                            C : Set ((i : ι) → X i)
                            J K : ι → Prop
                            inst✝¹ : ∀ (i : ι), T2Space (X i)
                            inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (X i)
                            hC : IsCompact C
                            ⊢ CompactSpace ↑C
                          -/
    @Profinite.of C _ (by rwa [← isCompact_iff_compactSpace]) _ _ ≅
                          /-
                            🎉 no goals
                          -/
    (Profinite.limitCone.{u, u} (indexFunctor hC)).pt :=
  asIso <| (Profinite.limitConeIsLimit.{u, u} _).lift (indexCone hC)


/-- The isomorphism of cones induced by `isoindexConeLift`. -/
noncomputable
def asLimitindexConeIso : indexCone hC ≅ Profinite.limitCone.{u, u} _ :=
  Limits.Cones.ext (isoindexConeLift hC) fun _ => rfl


/-- `indexCone` is a limit cone. -/
noncomputable
def indexCone_isLimit : CategoryTheory.Limits.IsLimit (indexCone hC) :=
  Limits.IsLimit.ofIsoLimit (Profinite.limitConeIsLimit _) (asLimitindexConeIso hC).symm


