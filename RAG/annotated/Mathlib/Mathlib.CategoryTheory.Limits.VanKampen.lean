/-- A natural transformation is equifibered if every commutative square of the following form is
a pullback.
```
F(X) → F(Y)
 ↓      ↓
G(X) → G(Y)
```
-/
def NatTrans.Equifibered {F G : J ⥤ C} (α : F ⟶ G) : Prop :=
  ∀ ⦃i j : J⦄ (f : i ⟶ j), IsPullback (F.map f) (α.app i) (α.app j) (G.map f)


theorem NatTrans.equifibered_of_isIso {F G : J ⥤ C} (α : F ⟶ G) [IsIso α] : Equifibered α :=
  fun _ _ f => IsPullback.of_vert_isIso ⟨NatTrans.naturality _ f⟩


theorem NatTrans.Equifibered.comp {F G H : J ⥤ C} {α : F ⟶ G} {β : G ⟶ H} (hα : Equifibered α)
    (hβ : Equifibered β) : Equifibered (α ≫ β) :=
  fun _ _ f => (hα f).paste_vert (hβ f)


theorem NatTrans.Equifibered.whiskerRight {F G : J ⥤ C} {α : F ⟶ G} (hα : Equifibered α)
    (H : C ⥤ D) [∀ (i j : J) (f : j ⟶ i), PreservesLimit (cospan (α.app i) (G.map f)) H] :
    Equifibered (whiskerRight α H) :=
  fun _ _ f => (hα f).map H


theorem NatTrans.Equifibered.whiskerLeft {K : Type*} [Category K]  {F G : J ⥤ C} {α : F ⟶ G}
    (hα : Equifibered α) (H : K ⥤ J) : Equifibered (whiskerLeft H α) :=
  fun _ _ f => hα (H.map f)


theorem mapPair_equifibered {F F' : Discrete WalkingPair ⥤ C} (α : F ⟶ F') :
    NatTrans.Equifibered α := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F F' : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.W …
    α : Quiver.Hom F F'
    ⊢ CategoryTheory.NatTrans.Equifibered α
  -/
  rintro ⟨⟨⟩⟩ ⟨j⟩ ⟨⟨rfl : _ = j⟩⟩
  all_goals
    dsimp; simp only [Discrete.functor_map_id]
    exact IsPullback.of_horiz_isIso ⟨by simp only [Category.comp_id, Category.id_comp]⟩


theorem NatTrans.equifibered_of_discrete {ι : Type*} {F G : Discrete ι ⥤ C}
    (α : F ⟶ G) : NatTrans.Equifibered α := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ι : Type u_3
    F G : CategoryTheory.Functor (CategoryTheory.Discrete ι) C
    α : Quiver.Hom F G
    ⊢ CategoryTheory.NatTrans.Equifibered α
  -/
  rintro ⟨i⟩ ⟨j⟩ ⟨⟨rfl : i = j⟩⟩
  /-
    case mk.mk.up.up
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ι : Type u_3
    F G : CategoryTheory.Functor (CategoryTheory.Discrete ι) C
    α : Quiver.Hom F G
    i : ι
    ⊢ CategoryTheory.IsPullback (F.map { down := { down := ⋯ } }) (α.app { as := i …
  -/
  simp only [Discrete.functor_map_id]
  /-
    case mk.mk.up.up
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ι : Type u_3
    F G : CategoryTheory.Functor (CategoryTheory.Discrete ι) C
    α : Quiver.Hom F G
    i : ι
    ⊢ CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.id (F.obj { as := i …
  -/
  exact IsPullback.of_horiz_isIso ⟨by rw [Category.id_comp, Category.comp_id]⟩
  /-
    🎉 no goals
  -/


/-- A (colimit) cocone over a diagram `F : J ⥤ C` is universal if it is stable under pullbacks. -/
def IsUniversalColimit {F : J ⥤ C} (c : Cocone F) : Prop :=
  ∀ ⦃F' : J ⥤ C⦄ (c' : Cocone F') (α : F' ⟶ F) (f : c'.pt ⟶ c.pt)
    (_ : α ≫ c.ι = c'.ι ≫ (Functor.const J).map f) (_ : NatTrans.Equifibered α),
    (∀ j : J, IsPullback (c'.ι.app j) (α.app j) f (c.ι.app j)) → Nonempty (IsColimit c')


/-- A (colimit) cocone over a diagram `F : J ⥤ C` is van Kampen if for every cocone `c'` over the
pullback of the diagram `F' : J ⥤ C'`, `c'` is colimiting iff `c'` is the pullback of `c`.

TODO: Show that this is iff the functor `C ⥤ Catᵒᵖ` sending `x` to `C/x` preserves it.
TODO: Show that this is iff the inclusion functor `C ⥤ Span(C)` preserves it.
-/
def IsVanKampenColimit {F : J ⥤ C} (c : Cocone F) : Prop :=
  ∀ ⦃F' : J ⥤ C⦄ (c' : Cocone F') (α : F' ⟶ F) (f : c'.pt ⟶ c.pt)
    (_ : α ≫ c.ι = c'.ι ≫ (Functor.const J).map f) (_ : NatTrans.Equifibered α),
    Nonempty (IsColimit c') ↔ ∀ j : J, IsPullback (c'.ι.app j) (α.app j) f (c.ι.app j)


theorem IsVanKampenColimit.isUniversal {F : J ⥤ C} {c : Cocone F} (H : IsVanKampenColimit c) :
    IsUniversalColimit c :=
  fun _ c' α f h hα => (H c' α f h hα).mpr


/-- A universal colimit is a colimit. -/
noncomputable def IsUniversalColimit.isColimit {F : J ⥤ C} {c : Cocone F}
    (h : IsUniversalColimit c) : IsColimit c := by
  refine ((h c (𝟙 F) (𝟙 c.pt : _) (by rw [Functor.map_id, Category.comp_id, Category.id_comp])
    (NatTrans.equifibered_of_isIso _)) fun j => ?_).some
  /-
    J : Type v'
    inst✝³ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    K : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.15863, u_1} K
    D : Type u_2
    inst✝ : CategoryTheory.Category.{?u.15870, u_2} D
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    h : CategoryTheory.IsUniversalColimit c
    j : J
    ⊢ CategoryTheory.IsPullback (c.ι.app j) ((CategoryTheory.CategoryStruct.id F). …
  -/
  haveI : IsIso (𝟙 c.pt) := inferInstance
  /-
    J : Type v'
    inst✝³ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    K : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.15863, u_1} K
    D : Type u_2
    inst✝ : CategoryTheory.Category.{?u.15870, u_2} D
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    h : CategoryTheory.IsUniversalColimit c
    j : J
    this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.id c.pt)
    ⊢ CategoryTheory.IsPullback (c.ι.app j) ((CategoryTheory.CategoryStruct.id F). …
  -/
  exact IsPullback.of_vert_isIso ⟨by erw [NatTrans.id_app, Category.comp_id, Category.id_comp]⟩
  /-
    🎉 no goals
  -/


/-- A van Kampen colimit is a colimit. -/
noncomputable def IsVanKampenColimit.isColimit {F : J ⥤ C} {c : Cocone F}
    (h : IsVanKampenColimit c) : IsColimit c :=
  h.isUniversal.isColimit


theorem IsInitial.isVanKampenColimit [HasStrictInitialObjects C] {X : C} (h : IsInitial X) :
    IsVanKampenColimit (asEmptyCocone X) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasStrictInitialObjects C
    X : C
    h : CategoryTheory.Limits.IsInitial X
    ⊢ CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.asEmptyCocone X)
  -/
  intro F' c' α f hf hα
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasStrictInitialObjects C
    X : C
    h : CategoryTheory.Limits.IsInitial X
    F' : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' (CategoryTheory.Functor.empty C)
    f : Quiver.Hom c'.pt (CategoryTheory.Limits.asEmptyCocone X).pt
    hf : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.asEmptyCo …
    hα : CategoryTheory.NatTrans.Equifibered α
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (∀ (j : CategoryTheory.D …
  -/
  have : F' = Functor.empty C := by apply Functor.hext <;> rintro ⟨⟨⟩⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasStrictInitialObjects C
    X : C
    h : CategoryTheory.Limits.IsInitial X
    F' : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' (CategoryTheory.Functor.empty C)
    f : Quiver.Hom c'.pt (CategoryTheory.Limits.asEmptyCocone X).pt
    hf : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.asEmptyCo …
    hα : CategoryTheory.NatTrans.Equifibered α
    this : Eq F' (CategoryTheory.Functor.empty C)
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (∀ (j : CategoryTheory.D …
  -/
  subst this
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasStrictInitialObjects C
    X : C
    h : CategoryTheory.Limits.IsInitial X
    c' : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty C)
    α : Quiver.Hom (CategoryTheory.Functor.empty C) (CategoryTheory.Functor.empty C)
    f : Quiver.Hom c'.pt (CategoryTheory.Limits.asEmptyCocone X).pt
    hf : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.asEmptyCo …
    hα : CategoryTheory.NatTrans.Equifibered α
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (∀ (j : CategoryTheory.D …
  -/
  haveI := h.isIso_to f
  refine ⟨by rintro _ ⟨⟨⟩⟩,
    fun _ => ⟨IsColimit.ofIsoColimit h (Cocones.ext (asIso f).symm <| by rintro ⟨⟨⟩⟩)⟩⟩


theorem IsUniversalColimit.of_iso {F : J ⥤ C} {c c' : Cocone F} (hc : IsUniversalColimit c)
    (e : c ≅ c') : IsUniversalColimit c' := by
  /-
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.IsUniversalColimit c
    e : CategoryTheory.Iso c c'
    ⊢ CategoryTheory.IsUniversalColimit c'
  -/
  intro F' c'' α f h hα H
  have : c'.ι ≫ (Functor.const J).map e.inv.hom = c.ι := by
    ext j
    exact e.inv.2 j
  /-
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.IsUniversalColimit c
    e : CategoryTheory.Iso c c'
    F' : CategoryTheory.Functor J C
    c'' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' F
    f : Quiver.Hom c''.pt c'.pt
    h : Eq (CategoryTheory.CategoryStruct.comp α c'.ι) (CategoryTheory.CategoryStr …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : J), CategoryTheory.IsPullback (c''.ι.app j) (α.app j) f (c'.ι.app j)
    this : Eq (CategoryTheory.CategoryStruct.comp c'.ι ((CategoryTheory.Functor.co …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c'')
  -/
  apply hc c'' α (f ≫ e.inv.1) (by rw [Functor.map_comp, ← reassoc_of% h, this]) hα
  /-
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.IsUniversalColimit c
    e : CategoryTheory.Iso c c'
    F' : CategoryTheory.Functor J C
    c'' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' F
    f : Quiver.Hom c''.pt c'.pt
    h : Eq (CategoryTheory.CategoryStruct.comp α c'.ι) (CategoryTheory.CategoryStr …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : J), CategoryTheory.IsPullback (c''.ι.app j) (α.app j) f (c'.ι.app j)
    this : Eq (CategoryTheory.CategoryStruct.comp c'.ι ((CategoryTheory.Functor.co …
    ⊢ ∀ (j : J), CategoryTheory.IsPullback (c''.ι.app j) (α.app j) (CategoryTheory …
  -/
  intro j
  /-
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.IsUniversalColimit c
    e : CategoryTheory.Iso c c'
    F' : CategoryTheory.Functor J C
    c'' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' F
    f : Quiver.Hom c''.pt c'.pt
    h : Eq (CategoryTheory.CategoryStruct.comp α c'.ι) (CategoryTheory.CategoryStr …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : J), CategoryTheory.IsPullback (c''.ι.app j) (α.app j) f (c'.ι.app j)
    this : Eq (CategoryTheory.CategoryStruct.comp c'.ι ((CategoryTheory.Functor.co …
    j : J
    ⊢ CategoryTheory.IsPullback (c''.ι.app j) (α.app j) (CategoryTheory.CategorySt …
  -/
  rw [← Category.comp_id (α.app j)]
  /-
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.IsUniversalColimit c
    e : CategoryTheory.Iso c c'
    F' : CategoryTheory.Functor J C
    c'' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' F
    f : Quiver.Hom c''.pt c'.pt
    h : Eq (CategoryTheory.CategoryStruct.comp α c'.ι) (CategoryTheory.CategoryStr …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : J), CategoryTheory.IsPullback (c''.ι.app j) (α.app j) f (c'.ι.app j)
    this : Eq (CategoryTheory.CategoryStruct.comp c'.ι ((CategoryTheory.Functor.co …
    j : J
    ⊢ CategoryTheory.IsPullback (c''.ι.app j) (CategoryTheory.CategoryStruct.comp  …
  -/
  have : IsIso e.inv.hom := Functor.map_isIso (Cocones.forget _) e.inv
  /-
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.IsUniversalColimit c
    e : CategoryTheory.Iso c c'
    F' : CategoryTheory.Functor J C
    c'' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' F
    f : Quiver.Hom c''.pt c'.pt
    h : Eq (CategoryTheory.CategoryStruct.comp α c'.ι) (CategoryTheory.CategoryStr …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : J), CategoryTheory.IsPullback (c''.ι.app j) (α.app j) f (c'.ι.app j)
    this✝ : Eq (CategoryTheory.CategoryStruct.comp c'.ι ((CategoryTheory.Functor.c …
    j : J
    this : CategoryTheory.IsIso e.inv.hom
    ⊢ CategoryTheory.IsPullback (c''.ι.app j) (CategoryTheory.CategoryStruct.comp  …
  -/
  exact (H j).paste_vert (IsPullback.of_vert_isIso ⟨by simp⟩)
  /-
    🎉 no goals
  -/


theorem IsVanKampenColimit.of_iso {F : J ⥤ C} {c c' : Cocone F} (H : IsVanKampenColimit c)
    (e : c ≅ c') : IsVanKampenColimit c' := by
  /-
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    H : CategoryTheory.IsVanKampenColimit c
    e : CategoryTheory.Iso c c'
    ⊢ CategoryTheory.IsVanKampenColimit c'
  -/
  intro F' c'' α f h hα
  have : c'.ι ≫ (Functor.const J).map e.inv.hom = c.ι := by
    ext j
    exact e.inv.2 j
  /-
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    H : CategoryTheory.IsVanKampenColimit c
    e : CategoryTheory.Iso c c'
    F' : CategoryTheory.Functor J C
    c'' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' F
    f : Quiver.Hom c''.pt c'.pt
    h : Eq (CategoryTheory.CategoryStruct.comp α c'.ι) (CategoryTheory.CategoryStr …
    hα : CategoryTheory.NatTrans.Equifibered α
    this : Eq (CategoryTheory.CategoryStruct.comp c'.ι ((CategoryTheory.Functor.co …
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c'')) (∀ (j : J), CategoryThe …
  -/
  rw [H c'' α (f ≫ e.inv.1) (by rw [Functor.map_comp, ← reassoc_of% h, this]) hα]
  /-
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    H : CategoryTheory.IsVanKampenColimit c
    e : CategoryTheory.Iso c c'
    F' : CategoryTheory.Functor J C
    c'' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' F
    f : Quiver.Hom c''.pt c'.pt
    h : Eq (CategoryTheory.CategoryStruct.comp α c'.ι) (CategoryTheory.CategoryStr …
    hα : CategoryTheory.NatTrans.Equifibered α
    this : Eq (CategoryTheory.CategoryStruct.comp c'.ι ((CategoryTheory.Functor.co …
    ⊢ Iff (∀ (j : J), CategoryTheory.IsPullback (c''.ι.app j) (α.app j) (CategoryT …
  -/
  apply forall_congr'
  /-
    case h
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    H : CategoryTheory.IsVanKampenColimit c
    e : CategoryTheory.Iso c c'
    F' : CategoryTheory.Functor J C
    c'' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' F
    f : Quiver.Hom c''.pt c'.pt
    h : Eq (CategoryTheory.CategoryStruct.comp α c'.ι) (CategoryTheory.CategoryStr …
    hα : CategoryTheory.NatTrans.Equifibered α
    this : Eq (CategoryTheory.CategoryStruct.comp c'.ι ((CategoryTheory.Functor.co …
    ⊢ ∀ (a : J), Iff (CategoryTheory.IsPullback (c''.ι.app a) (α.app a) (CategoryT …
  -/
  intro j
  /-
    case h
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    H : CategoryTheory.IsVanKampenColimit c
    e : CategoryTheory.Iso c c'
    F' : CategoryTheory.Functor J C
    c'' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' F
    f : Quiver.Hom c''.pt c'.pt
    h : Eq (CategoryTheory.CategoryStruct.comp α c'.ι) (CategoryTheory.CategoryStr …
    hα : CategoryTheory.NatTrans.Equifibered α
    this : Eq (CategoryTheory.CategoryStruct.comp c'.ι ((CategoryTheory.Functor.co …
    j : J
    ⊢ Iff (CategoryTheory.IsPullback (c''.ι.app j) (α.app j) (CategoryTheory.Categ …
  -/
  conv_lhs => rw [← Category.comp_id (α.app j)]
  /-
    case h
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    H : CategoryTheory.IsVanKampenColimit c
    e : CategoryTheory.Iso c c'
    F' : CategoryTheory.Functor J C
    c'' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' F
    f : Quiver.Hom c''.pt c'.pt
    h : Eq (CategoryTheory.CategoryStruct.comp α c'.ι) (CategoryTheory.CategoryStr …
    hα : CategoryTheory.NatTrans.Equifibered α
    this : Eq (CategoryTheory.CategoryStruct.comp c'.ι ((CategoryTheory.Functor.co …
    j : J
    ⊢ Iff (CategoryTheory.IsPullback (c''.ι.app j) (CategoryTheory.CategoryStruct. …
  -/
  haveI : IsIso e.inv.hom := Functor.map_isIso (Cocones.forget _) e.inv
  /-
    case h
    J : Type v'
    inst✝¹ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    c c' : CategoryTheory.Limits.Cocone F
    H : CategoryTheory.IsVanKampenColimit c
    e : CategoryTheory.Iso c c'
    F' : CategoryTheory.Functor J C
    c'' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' F
    f : Quiver.Hom c''.pt c'.pt
    h : Eq (CategoryTheory.CategoryStruct.comp α c'.ι) (CategoryTheory.CategoryStr …
    hα : CategoryTheory.NatTrans.Equifibered α
    this✝ : Eq (CategoryTheory.CategoryStruct.comp c'.ι ((CategoryTheory.Functor.c …
    j : J
    this : CategoryTheory.IsIso e.inv.hom
    ⊢ Iff (CategoryTheory.IsPullback (c''.ι.app j) (CategoryTheory.CategoryStruct. …
  -/
  exact (IsPullback.of_vert_isIso ⟨by simp⟩).paste_vert_iff (NatTrans.congr_app h j).symm
  /-
    🎉 no goals
  -/


theorem IsVanKampenColimit.precompose_isIso {F G : J ⥤ C} (α : F ⟶ G) [IsIso α]
    {c : Cocone G} (hc : IsVanKampenColimit c) :
    IsVanKampenColimit ((Cocones.precompose α).obj c) := by
  /-
    J : Type v'
    inst✝² : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso α
    c : CategoryTheory.Limits.Cocone G
    hc : CategoryTheory.IsVanKampenColimit c
    ⊢ CategoryTheory.IsVanKampenColimit ((CategoryTheory.Limits.Cocones.precompose …
  -/
  intros F' c' α' f e hα
  refine (hc c' (α' ≫ α) f ((Category.assoc _ _ _).trans e)
    (hα.comp (NatTrans.equifibered_of_isIso _))).trans ?_
  /-
    J : Type v'
    inst✝² : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso α
    c : CategoryTheory.Limits.Cocone G
    hc : CategoryTheory.IsVanKampenColimit c
    F' : CategoryTheory.Functor J C
    c' : CategoryTheory.Limits.Cocone F'
    α' : Quiver.Hom F' F
    f : Quiver.Hom c'.pt ((CategoryTheory.Limits.Cocones.precompose α).obj c).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α' ((CategoryTheory.Limits.Cocones. …
    hα : CategoryTheory.NatTrans.Equifibered α'
    ⊢ Iff (∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) ((CategoryTheory.Cate …
  -/
  apply forall_congr'
  /-
    case h
    J : Type v'
    inst✝² : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso α
    c : CategoryTheory.Limits.Cocone G
    hc : CategoryTheory.IsVanKampenColimit c
    F' : CategoryTheory.Functor J C
    c' : CategoryTheory.Limits.Cocone F'
    α' : Quiver.Hom F' F
    f : Quiver.Hom c'.pt ((CategoryTheory.Limits.Cocones.precompose α).obj c).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α' ((CategoryTheory.Limits.Cocones. …
    hα : CategoryTheory.NatTrans.Equifibered α'
    ⊢ ∀ (a : J), Iff (CategoryTheory.IsPullback (c'.ι.app a) ((CategoryTheory.Cate …
  -/
  intro j
  simp only [Functor.const_obj_obj, NatTrans.comp_app,
    Cocones.precompose_obj_pt, Cocones.precompose_obj_ι]
  have : IsPullback (α.app j ≫ c.ι.app j) (α.app j) (𝟙 _) (c.ι.app j) :=
    IsPullback.of_vert_isIso ⟨Category.comp_id _⟩
  /-
    case h
    J : Type v'
    inst✝² : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso α
    c : CategoryTheory.Limits.Cocone G
    hc : CategoryTheory.IsVanKampenColimit c
    F' : CategoryTheory.Functor J C
    c' : CategoryTheory.Limits.Cocone F'
    α' : Quiver.Hom F' F
    f : Quiver.Hom c'.pt ((CategoryTheory.Limits.Cocones.precompose α).obj c).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α' ((CategoryTheory.Limits.Cocones. …
    hα : CategoryTheory.NatTrans.Equifibered α'
    j : J
    this : CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.comp (α.app j) …
    ⊢ Iff (CategoryTheory.IsPullback (c'.ι.app j) (CategoryTheory.CategoryStruct.c …
  -/
  rw [← IsPullback.paste_vert_iff this _, Category.comp_id]
  /-
    J : Type v'
    inst✝² : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso α
    c : CategoryTheory.Limits.Cocone G
    hc : CategoryTheory.IsVanKampenColimit c
    F' : CategoryTheory.Functor J C
    c' : CategoryTheory.Limits.Cocone F'
    α' : Quiver.Hom F' F
    f : Quiver.Hom c'.pt ((CategoryTheory.Limits.Cocones.precompose α).obj c).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α' ((CategoryTheory.Limits.Cocones. …
    hα : CategoryTheory.NatTrans.Equifibered α'
    j : J
    this : CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.comp (α.app j) …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (c'.ι.app j) f) (CategoryTheory.Categ …
  -/
  exact (congr_app e j).symm
  /-
    🎉 no goals
  -/


theorem IsUniversalColimit.precompose_isIso {F G : J ⥤ C} (α : F ⟶ G) [IsIso α]
    {c : Cocone G} (hc : IsUniversalColimit c) :
    IsUniversalColimit ((Cocones.precompose α).obj c) := by
  /-
    J : Type v'
    inst✝² : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso α
    c : CategoryTheory.Limits.Cocone G
    hc : CategoryTheory.IsUniversalColimit c
    ⊢ CategoryTheory.IsUniversalColimit ((CategoryTheory.Limits.Cocones.precompose …
  -/
  intros F' c' α' f e hα H
  apply (hc c' (α' ≫ α) f ((Category.assoc _ _ _).trans e)
    (hα.comp (NatTrans.equifibered_of_isIso _)))
  /-
    J : Type v'
    inst✝² : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso α
    c : CategoryTheory.Limits.Cocone G
    hc : CategoryTheory.IsUniversalColimit c
    F' : CategoryTheory.Functor J C
    c' : CategoryTheory.Limits.Cocone F'
    α' : Quiver.Hom F' F
    f : Quiver.Hom c'.pt ((CategoryTheory.Limits.Cocones.precompose α).obj c).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α' ((CategoryTheory.Limits.Cocones. …
    hα : CategoryTheory.NatTrans.Equifibered α'
    H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α'.app j) f (((Category …
    ⊢ ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) ((CategoryTheory.CategoryS …
  -/
  intro j
  simp only [Functor.const_obj_obj, NatTrans.comp_app,
    Cocones.precompose_obj_pt, Cocones.precompose_obj_ι]
  /-
    J : Type v'
    inst✝² : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso α
    c : CategoryTheory.Limits.Cocone G
    hc : CategoryTheory.IsUniversalColimit c
    F' : CategoryTheory.Functor J C
    c' : CategoryTheory.Limits.Cocone F'
    α' : Quiver.Hom F' F
    f : Quiver.Hom c'.pt ((CategoryTheory.Limits.Cocones.precompose α).obj c).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α' ((CategoryTheory.Limits.Cocones. …
    hα : CategoryTheory.NatTrans.Equifibered α'
    H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α'.app j) f (((Category …
    j : J
    ⊢ CategoryTheory.IsPullback (c'.ι.app j) (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [← Category.comp_id f]
  /-
    J : Type v'
    inst✝² : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.IsIso α
    c : CategoryTheory.Limits.Cocone G
    hc : CategoryTheory.IsUniversalColimit c
    F' : CategoryTheory.Functor J C
    c' : CategoryTheory.Limits.Cocone F'
    α' : Quiver.Hom F' F
    f : Quiver.Hom c'.pt ((CategoryTheory.Limits.Cocones.precompose α).obj c).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α' ((CategoryTheory.Limits.Cocones. …
    hα : CategoryTheory.NatTrans.Equifibered α'
    H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α'.app j) f (((Category …
    j : J
    ⊢ CategoryTheory.IsPullback (c'.ι.app j) (CategoryTheory.CategoryStruct.comp ( …
  -/
  exact (H j).paste_vert (IsPullback.of_vert_isIso ⟨Category.comp_id _⟩)
  /-
    🎉 no goals
  -/


theorem IsVanKampenColimit.precompose_isIso_iff {F G : J ⥤ C} (α : F ⟶ G) [IsIso α]
    {c : Cocone G} : IsVanKampenColimit ((Cocones.precompose α).obj c) ↔ IsVanKampenColimit c :=
  ⟨fun hc ↦ IsVanKampenColimit.of_iso (IsVanKampenColimit.precompose_isIso (inv α) hc)
                                  /-
                                    J : Type v'
                                    inst✝² : CategoryTheory.Category.{u', v'} J
                                    C : Type u
                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                    F G : CategoryTheory.Functor J C
                                    α : Quiver.Hom F G
                                    inst✝ : CategoryTheory.IsIso α
                                    c : CategoryTheory.Limits.Cocone G
                                    hc : CategoryTheory.IsVanKampenColimit ((CategoryTheory.Limits.Cocones.precomp …
                                    ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.C …
                                  -/
    (Cocones.ext (Iso.refl _) (by simp)),
                                  /-
                                    🎉 no goals
                                  -/
    IsVanKampenColimit.precompose_isIso α⟩


theorem IsUniversalColimit.of_mapCocone (G : C ⥤ D) {F : J ⥤ C} {c : Cocone F}
    [PreservesLimitsOfShape WalkingCospan G] [ReflectsColimitsOfShape J G]
    (hc : IsUniversalColimit (G.mapCocone c)) : IsUniversalColimit c :=
  fun F' c' α f h hα H ↦
    ⟨isColimitOfReflects _ (hc (G.mapCocone c') (whiskerRight α G) (G.map f)
        /-
          J : Type v'
          inst✝⁴ : CategoryTheory.Category.{u', v'} J
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{u_3, u_2} D
          G : CategoryTheory.Functor C D
          F : CategoryTheory.Functor J C
          c : CategoryTheory.Limits.Cocone F
          inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
          inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape J G
          hc : CategoryTheory.IsUniversalColimit (G.mapCocone c)
          F' : CategoryTheory.Functor J C
          c' : CategoryTheory.Limits.Cocone F'
          α : Quiver.Hom F' F
          f : Quiver.Hom c'.pt c.pt
          h : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStru …
          hα : CategoryTheory.NatTrans.Equifibered α
          H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f (c.ι.app j)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight α G) (G. …
        -/
    (by ext j; simpa using G.congr_map (NatTrans.congr_app h j))
               /-
                 🎉 no goals
               -/
    (hα.whiskerRight G) (fun j ↦ (H j).map G)).some⟩


theorem IsVanKampenColimit.of_mapCocone (G : C ⥤ D) {F : J ⥤ C} {c : Cocone F}
    [∀ (i j : J) (X : C) (f : X ⟶ F.obj j) (g : i ⟶ j), PreservesLimit (cospan f (F.map g)) G]
    [∀ (i : J) (X : C) (f : X ⟶ c.pt), PreservesLimit (cospan f (c.ι.app i)) G]
    [ReflectsLimitsOfShape WalkingCospan G]
    [PreservesColimitsOfShape J G]
    [ReflectsColimitsOfShape J G]
    (H : IsVanKampenColimit (G.mapCocone c)) : IsVanKampenColimit c := by
  /-
    J : Type v'
    inst✝⁷ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    inst✝⁴ : ∀ (i j : J) (X : C) (f : Quiver.Hom X (F.obj j)) (g : Quiver.Hom i j) …
    inst✝³ : ∀ (i : J) (X : C) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Pres …
    inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape J G
    inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape J G
    H : CategoryTheory.IsVanKampenColimit (G.mapCocone c)
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  intro F' c' α f h hα
  refine (Iff.trans ?_ (H (G.mapCocone c') (whiskerRight α G) (G.map f)
      (by ext j; simpa using G.congr_map (NatTrans.congr_app h j))
      (hα.whiskerRight G))).trans (forall_congr' fun j => ?_)
    /-
      case refine_1
      J : Type v'
      inst✝⁷ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      inst✝⁴ : ∀ (i j : J) (X : C) (f : Quiver.Hom X (F.obj j)) (g : Quiver.Hom i j) …
      inst✝³ : ∀ (i : J) (X : C) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Pres …
      inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape J G
      inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape J G
      H : CategoryTheory.IsVanKampenColimit (G.mapCocone c)
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' F
      f : Quiver.Hom c'.pt c.pt
      h : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStru …
      hα : CategoryTheory.NatTrans.Equifibered α
      ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (Nonempty (CategoryTheor …
    -/
  · exact ⟨fun h => ⟨isColimitOfPreserves G h.some⟩, fun h => ⟨isColimitOfReflects G h.some⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      J : Type v'
      inst✝⁷ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      inst✝⁴ : ∀ (i j : J) (X : C) (f : Quiver.Hom X (F.obj j)) (g : Quiver.Hom i j) …
      inst✝³ : ∀ (i : J) (X : C) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Pres …
      inst✝² : CategoryTheory.Limits.ReflectsLimitsOfShape CategoryTheory.Limits.Wal …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape J G
      inst✝ : CategoryTheory.Limits.ReflectsColimitsOfShape J G
      H : CategoryTheory.IsVanKampenColimit (G.mapCocone c)
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' F
      f : Quiver.Hom c'.pt c.pt
      h : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStru …
      hα : CategoryTheory.NatTrans.Equifibered α
      j : J
      ⊢ Iff (CategoryTheory.IsPullback ((G.mapCocone c').ι.app j) ((CategoryTheory.w …
    -/
  · exact IsPullback.map_iff G (NatTrans.congr_app h.symm j)
    /-
      🎉 no goals
    -/


theorem IsVanKampenColimit.mapCocone_iff (G : C ⥤ D) {F : J ⥤ C} {c : Cocone F}
    [G.IsEquivalence] : IsVanKampenColimit (G.mapCocone c) ↔ IsVanKampenColimit c :=
  ⟨IsVanKampenColimit.of_mapCocone G, fun hc ↦ by
    /-
      J : Type v'
      inst✝³ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      inst✝ : G.IsEquivalence
      hc : CategoryTheory.IsVanKampenColimit c
      ⊢ CategoryTheory.IsVanKampenColimit (G.mapCocone c)
    -/
    let e : F ⋙ G ⋙ Functor.inv G ≅ F := NatIso.hcomp (Iso.refl F) G.asEquivalence.unitIso.symm
    /-
      J : Type v'
      inst✝³ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      inst✝ : G.IsEquivalence
      hc : CategoryTheory.IsVanKampenColimit c
      e : CategoryTheory.Iso (F.comp (G.comp G.inv)) F := CategoryTheory.NatIso.hcom …
      ⊢ CategoryTheory.IsVanKampenColimit (G.mapCocone c)
    -/
    apply IsVanKampenColimit.of_mapCocone G.inv
    /-
      J : Type v'
      inst✝³ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      inst✝ : G.IsEquivalence
      hc : CategoryTheory.IsVanKampenColimit c
      e : CategoryTheory.Iso (F.comp (G.comp G.inv)) F := CategoryTheory.NatIso.hcom …
      ⊢ CategoryTheory.IsVanKampenColimit (G.inv.mapCocone (G.mapCocone c))
    -/
    apply (IsVanKampenColimit.precompose_isIso_iff e.inv).mp
    /-
      J : Type v'
      inst✝³ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      inst✝ : G.IsEquivalence
      hc : CategoryTheory.IsVanKampenColimit c
      e : CategoryTheory.Iso (F.comp (G.comp G.inv)) F := CategoryTheory.NatIso.hcom …
      ⊢ CategoryTheory.IsVanKampenColimit ((CategoryTheory.Limits.Cocones.precompose …
    -/
    exact hc.of_iso (Cocones.ext (G.asEquivalence.unitIso.app c.pt) (fun j => (by simp [e])))⟩
    /-
      🎉 no goals
    -/


theorem IsUniversalColimit.whiskerEquivalence {K : Type*} [Category K] (e : J ≌ K)
    {F : K ⥤ C} {c : Cocone F} (hc : IsUniversalColimit c) :
    IsUniversalColimit (c.whisker e.functor) := by
  /-
    J : Type v'
    inst✝² : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    K : Type u_3
    inst✝ : CategoryTheory.Category.{u_4, u_3} K
    e : CategoryTheory.Equivalence J K
    F : CategoryTheory.Functor K C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.IsUniversalColimit c
    ⊢ CategoryTheory.IsUniversalColimit (CategoryTheory.Limits.Cocone.whisker e.fu …
  -/
  intro F' c' α f e' hα H
  convert hc (c'.whisker e.inverse) (whiskerLeft e.inverse α ≫ (e.invFunIdAssoc F).hom) f ?_
    ((hα.whiskerLeft _).comp (NatTrans.equifibered_of_isIso _)) ?_ using 1
    /-
      case a
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsUniversalColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((CategoryTh …
      ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (Nonempty (CategoryTheor …
    -/
  · exact (IsColimit.whiskerEquivalenceEquiv e.symm).nonempty_congr
    /-
      🎉 no goals
    -/
    /-
      case convert_1
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsUniversalColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((CategoryTh …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · convert congr_arg (whiskerLeft e.inverse) e'
    /-
      case h.e'_2.h
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsUniversalColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((CategoryTh …
      e_1✝ : Eq (Quiver.Hom (e.inverse.comp F') ((CategoryTheory.Functor.const K).ob …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    ext
    /-
      case h.e'_2.h.w.h
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsUniversalColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((CategoryTh …
      e_1✝ : Eq (Quiver.Hom (e.inverse.comp F') ((CategoryTheory.Functor.const K).ob …
      x✝ : K
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsUniversalColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((CategoryTh …
      ⊢ ∀ (j : K), CategoryTheory.IsPullback ((CategoryTheory.Limits.Cocone.whisker  …
    -/
  · intro k
    /-
      case convert_2
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsUniversalColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((CategoryTh …
      k : K
      ⊢ CategoryTheory.IsPullback ((CategoryTheory.Limits.Cocone.whisker e.inverse c …
    -/
    rw [← Category.comp_id f]
    /-
      case convert_2
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsUniversalColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((CategoryTh …
      k : K
      ⊢ CategoryTheory.IsPullback ((CategoryTheory.Limits.Cocone.whisker e.inverse c …
    -/
    refine (H (e.inverse.obj k)).paste_vert ?_
    /-
      case convert_2
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsUniversalColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((CategoryTh …
      k : K
      ⊢ CategoryTheory.IsPullback ((CategoryTheory.Limits.Cocone.whisker e.functor c …
    -/
    have : IsIso (𝟙 (Cocone.whisker e.functor c).pt) := inferInstance
    /-
      case convert_2
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsUniversalColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((CategoryTh …
      k : K
      this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.id (CategoryTheory. …
      ⊢ CategoryTheory.IsPullback ((CategoryTheory.Limits.Cocone.whisker e.functor c …
    -/
    exact IsPullback.of_vert_isIso ⟨by simp⟩
    /-
      🎉 no goals
    -/


theorem IsUniversalColimit.whiskerEquivalence_iff {K : Type*} [Category K] (e : J ≌ K)
    {F : K ⥤ C} {c : Cocone F} :
    IsUniversalColimit (c.whisker e.functor) ↔ IsUniversalColimit c :=
  ⟨fun hc ↦ ((hc.whiskerEquivalence e.symm).precompose_isIso (e.invFunIdAssoc F).inv).of_iso
                                    /-
                                      J : Type v'
                                      inst✝² : CategoryTheory.Category.{u', v'} J
                                      C : Type u
                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                      K : Type u_3
                                      inst✝ : CategoryTheory.Category.{u_4, u_3} K
                                      e : CategoryTheory.Equivalence J K
                                      F : CategoryTheory.Functor K C
                                      c : CategoryTheory.Limits.Cocone F
                                      hc : CategoryTheory.IsUniversalColimit (CategoryTheory.Limits.Cocone.whisker e …
                                      ⊢ ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.C …
                                    -/
      (Cocones.ext (Iso.refl _) (by simp)), IsUniversalColimit.whiskerEquivalence e⟩
                                    /-
                                      🎉 no goals
                                    -/


theorem IsVanKampenColimit.whiskerEquivalence {K : Type*} [Category K] (e : J ≌ K)
    {F : K ⥤ C} {c : Cocone F} (hc : IsVanKampenColimit c) :
    IsVanKampenColimit (c.whisker e.functor) := by
  /-
    J : Type v'
    inst✝² : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    K : Type u_3
    inst✝ : CategoryTheory.Category.{u_4, u_3} K
    e : CategoryTheory.Equivalence J K
    F : CategoryTheory.Functor K C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.IsVanKampenColimit c
    ⊢ CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.Cocone.whisker e.fu …
  -/
  intro F' c' α f e' hα
  convert hc (c'.whisker e.inverse) (whiskerLeft e.inverse α ≫ (e.invFunIdAssoc F).hom) f ?_
    ((hα.whiskerLeft _).comp (NatTrans.equifibered_of_isIso _)) using 1
    /-
      case h.e'_1.a
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsVanKampenColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (Nonempty (CategoryTheor …
    -/
  · exact (IsColimit.whiskerEquivalenceEquiv e.symm).nonempty_congr
    /-
      🎉 no goals
    -/
  · simp only [Functor.const_obj_obj, Functor.comp_obj, Cocone.whisker_pt, Cocone.whisker_ι,
      whiskerLeft_app, NatTrans.comp_app, Equivalence.invFunIdAssoc_hom_app, Functor.id_obj]
    /-
      case h.e'_2.a
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsVanKampenColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      ⊢ Iff (∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f (c.ι.app  …
    -/
    constructor
      /-
        case h.e'_2.a.mp
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        K : Type u_3
        inst✝ : CategoryTheory.Category.{u_4, u_3} K
        e : CategoryTheory.Equivalence J K
        F : CategoryTheory.Functor K C
        c : CategoryTheory.Limits.Cocone F
        hc : CategoryTheory.IsVanKampenColimit c
        F' : CategoryTheory.Functor J C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (e.functor.comp F)
        f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
        e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
        hα : CategoryTheory.NatTrans.Equifibered α
        ⊢ (∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f (c.ι.app (e.f …
      -/
    · intro H k
      /-
        case h.e'_2.a.mp
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        K : Type u_3
        inst✝ : CategoryTheory.Category.{u_4, u_3} K
        e : CategoryTheory.Equivalence J K
        F : CategoryTheory.Functor K C
        c : CategoryTheory.Limits.Cocone F
        hc : CategoryTheory.IsVanKampenColimit c
        F' : CategoryTheory.Functor J C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (e.functor.comp F)
        f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
        e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f (c.ι.app (e. …
        k : K
        ⊢ CategoryTheory.IsPullback (c'.ι.app (e.inverse.obj k)) (CategoryTheory.Categ …
      -/
      rw [← Category.comp_id f]
      /-
        case h.e'_2.a.mp
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        K : Type u_3
        inst✝ : CategoryTheory.Category.{u_4, u_3} K
        e : CategoryTheory.Equivalence J K
        F : CategoryTheory.Functor K C
        c : CategoryTheory.Limits.Cocone F
        hc : CategoryTheory.IsVanKampenColimit c
        F' : CategoryTheory.Functor J C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (e.functor.comp F)
        f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
        e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f (c.ι.app (e. …
        k : K
        ⊢ CategoryTheory.IsPullback (c'.ι.app (e.inverse.obj k)) (CategoryTheory.Categ …
      -/
      refine (H (e.inverse.obj k)).paste_vert ?_
      /-
        case h.e'_2.a.mp
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        K : Type u_3
        inst✝ : CategoryTheory.Category.{u_4, u_3} K
        e : CategoryTheory.Equivalence J K
        F : CategoryTheory.Functor K C
        c : CategoryTheory.Limits.Cocone F
        hc : CategoryTheory.IsVanKampenColimit c
        F' : CategoryTheory.Functor J C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (e.functor.comp F)
        f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
        e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f (c.ι.app (e. …
        k : K
        ⊢ CategoryTheory.IsPullback (c.ι.app (e.functor.obj (e.inverse.obj k))) (F.map …
      -/
      have : IsIso (𝟙 (Cocone.whisker e.functor c).pt) := inferInstance
      /-
        case h.e'_2.a.mp
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        K : Type u_3
        inst✝ : CategoryTheory.Category.{u_4, u_3} K
        e : CategoryTheory.Equivalence J K
        F : CategoryTheory.Functor K C
        c : CategoryTheory.Limits.Cocone F
        hc : CategoryTheory.IsVanKampenColimit c
        F' : CategoryTheory.Functor J C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (e.functor.comp F)
        f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
        e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f (c.ι.app (e. …
        k : K
        this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.id (CategoryTheory. …
        ⊢ CategoryTheory.IsPullback (c.ι.app (e.functor.obj (e.inverse.obj k))) (F.map …
      -/
      exact IsPullback.of_vert_isIso ⟨by simp⟩
      /-
        🎉 no goals
      -/
      /-
        case h.e'_2.a.mpr
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        K : Type u_3
        inst✝ : CategoryTheory.Category.{u_4, u_3} K
        e : CategoryTheory.Equivalence J K
        F : CategoryTheory.Functor K C
        c : CategoryTheory.Limits.Cocone F
        hc : CategoryTheory.IsVanKampenColimit c
        F' : CategoryTheory.Functor J C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (e.functor.comp F)
        f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
        e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
        hα : CategoryTheory.NatTrans.Equifibered α
        ⊢ (∀ (j : K), CategoryTheory.IsPullback (c'.ι.app (e.inverse.obj j)) (Category …
      -/
    · intro H j
      have : α.app j
          = F'.map (e.unit.app _) ≫ α.app _ ≫ F.map (e.counit.app (e.functor.obj j)) := by
        simp [← Functor.map_comp]
      /-
        case h.e'_2.a.mpr
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        K : Type u_3
        inst✝ : CategoryTheory.Category.{u_4, u_3} K
        e : CategoryTheory.Equivalence J K
        F : CategoryTheory.Functor K C
        c : CategoryTheory.Limits.Cocone F
        hc : CategoryTheory.IsVanKampenColimit c
        F' : CategoryTheory.Functor J C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (e.functor.comp F)
        f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
        e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : K), CategoryTheory.IsPullback (c'.ι.app (e.inverse.obj j)) (Categor …
        j : J
        this : Eq (α.app j) (CategoryTheory.CategoryStruct.comp (F'.map (e.unit.app j) …
        ⊢ CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f (c.ι.app (e.functor.obj j))
      -/
      rw [← Category.id_comp f, this]
      /-
        case h.e'_2.a.mpr
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        K : Type u_3
        inst✝ : CategoryTheory.Category.{u_4, u_3} K
        e : CategoryTheory.Equivalence J K
        F : CategoryTheory.Functor K C
        c : CategoryTheory.Limits.Cocone F
        hc : CategoryTheory.IsVanKampenColimit c
        F' : CategoryTheory.Functor J C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (e.functor.comp F)
        f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
        e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : K), CategoryTheory.IsPullback (c'.ι.app (e.inverse.obj j)) (Categor …
        j : J
        this : Eq (α.app j) (CategoryTheory.CategoryStruct.comp (F'.map (e.unit.app j) …
        ⊢ CategoryTheory.IsPullback (c'.ι.app j) (CategoryTheory.CategoryStruct.comp ( …
      -/
      refine IsPullback.paste_vert ?_ (H (e.functor.obj j))
      /-
        case h.e'_2.a.mpr
        J : Type v'
        inst✝² : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        K : Type u_3
        inst✝ : CategoryTheory.Category.{u_4, u_3} K
        e : CategoryTheory.Equivalence J K
        F : CategoryTheory.Functor K C
        c : CategoryTheory.Limits.Cocone F
        hc : CategoryTheory.IsVanKampenColimit c
        F' : CategoryTheory.Functor J C
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' (e.functor.comp F)
        f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
        e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : K), CategoryTheory.IsPullback (c'.ι.app (e.inverse.obj j)) (Categor …
        j : J
        this : Eq (α.app j) (CategoryTheory.CategoryStruct.comp (F'.map (e.unit.app j) …
        ⊢ CategoryTheory.IsPullback (c'.ι.app j) (F'.map (e.unit.app j)) (CategoryTheo …
      -/
      exact IsPullback.of_vert_isIso ⟨by simp⟩
      /-
        🎉 no goals
      -/
    /-
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsVanKampenColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · ext k
    /-
      case w.h
      J : Type v'
      inst✝² : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      K : Type u_3
      inst✝ : CategoryTheory.Category.{u_4, u_3} K
      e : CategoryTheory.Equivalence J K
      F : CategoryTheory.Functor K C
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.IsVanKampenColimit c
      F' : CategoryTheory.Functor J C
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (e.functor.comp F)
      f : Quiver.Hom c'.pt (CategoryTheory.Limits.Cocone.whisker e.functor c).pt
      e' : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.Limits.Cocone.wh …
      hα : CategoryTheory.NatTrans.Equifibered α
      k : K
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
    simpa using congr_app e' (e.inverse.obj k)
    /-
      🎉 no goals
    -/


theorem IsVanKampenColimit.whiskerEquivalence_iff {K : Type*} [Category K] (e : J ≌ K)
    {F : K ⥤ C} {c : Cocone F} :
    IsVanKampenColimit (c.whisker e.functor) ↔ IsVanKampenColimit c :=
  ⟨fun hc ↦ ((hc.whiskerEquivalence e.symm).precompose_isIso (e.invFunIdAssoc F).inv).of_iso
                                    /-
                                      J : Type v'
                                      inst✝² : CategoryTheory.Category.{u', v'} J
                                      C : Type u
                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                      K : Type u_3
                                      inst✝ : CategoryTheory.Category.{u_4, u_3} K
                                      e : CategoryTheory.Equivalence J K
                                      F : CategoryTheory.Functor K C
                                      c : CategoryTheory.Limits.Cocone F
                                      hc : CategoryTheory.IsVanKampenColimit (CategoryTheory.Limits.Cocone.whisker e …
                                      ⊢ ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.C …
                                    -/
      (Cocones.ext (Iso.refl _) (by simp)), IsVanKampenColimit.whiskerEquivalence e⟩
                                    /-
                                      🎉 no goals
                                    -/


theorem isVanKampenColimit_of_evaluation [HasPullbacks D] [HasColimitsOfShape J D] (F : J ⥤ C ⥤ D)
    (c : Cocone F) (hc : ∀ x : C, IsVanKampenColimit (((evaluation C D).obj x).mapCocone c)) :
    IsVanKampenColimit c := by
  /-
    J : Type v'
    inst✝⁴ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : CategoryTheory.Limits.HasPullbacks D
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
    F : CategoryTheory.Functor J (CategoryTheory.Functor C D)
    c : CategoryTheory.Limits.Cocone F
    hc : ∀ (x : C), CategoryTheory.IsVanKampenColimit (((CategoryTheory.evaluation …
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  intro F' c' α f e hα
  have := fun x => hc x (((evaluation C D).obj x).mapCocone c') (whiskerRight α _)
      (((evaluation C D).obj x).map f)
      (by
        ext y
        dsimp
        exact NatTrans.congr_app (NatTrans.congr_app e y) x)
      (hα.whiskerRight _)
  /-
    J : Type v'
    inst✝⁴ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : CategoryTheory.Limits.HasPullbacks D
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
    F : CategoryTheory.Functor J (CategoryTheory.Functor C D)
    c : CategoryTheory.Limits.Cocone F
    hc : ∀ (x : C), CategoryTheory.IsVanKampenColimit (((CategoryTheory.evaluation …
    F' : CategoryTheory.Functor J (CategoryTheory.Functor C D)
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' F
    f : Quiver.Hom c'.pt c.pt
    e : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStru …
    hα : CategoryTheory.NatTrans.Equifibered α
    this : ∀ (x : C), Iff (Nonempty (CategoryTheory.Limits.IsColimit (((CategoryTh …
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (∀ (j : J), CategoryTheo …
  -/
  constructor
    /-
      case mp
      J : Type v'
      inst✝⁴ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Limits.HasPullbacks D
      inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
      F : CategoryTheory.Functor J (CategoryTheory.Functor C D)
      c : CategoryTheory.Limits.Cocone F
      hc : ∀ (x : C), CategoryTheory.IsVanKampenColimit (((CategoryTheory.evaluation …
      F' : CategoryTheory.Functor J (CategoryTheory.Functor C D)
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' F
      f : Quiver.Hom c'.pt c.pt
      e : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStru …
      hα : CategoryTheory.NatTrans.Equifibered α
      this : ∀ (x : C), Iff (Nonempty (CategoryTheory.Limits.IsColimit (((CategoryTh …
      ⊢ Nonempty (CategoryTheory.Limits.IsColimit c') → ∀ (j : J), CategoryTheory.Is …
    -/
  · rintro ⟨hc'⟩ j
    /-
      case mp.intro
      J : Type v'
      inst✝⁴ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Limits.HasPullbacks D
      inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
      F : CategoryTheory.Functor J (CategoryTheory.Functor C D)
      c : CategoryTheory.Limits.Cocone F
      hc : ∀ (x : C), CategoryTheory.IsVanKampenColimit (((CategoryTheory.evaluation …
      F' : CategoryTheory.Functor J (CategoryTheory.Functor C D)
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' F
      f : Quiver.Hom c'.pt c.pt
      e : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStru …
      hα : CategoryTheory.NatTrans.Equifibered α
      this : ∀ (x : C), Iff (Nonempty (CategoryTheory.Limits.IsColimit (((CategoryTh …
      hc' : CategoryTheory.Limits.IsColimit c'
      j : J
      ⊢ CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f (c.ι.app j)
    -/
    refine ⟨⟨(NatTrans.congr_app e j).symm⟩, ⟨evaluationJointlyReflectsLimits _ ?_⟩⟩
    /-
      case mp.intro
      J : Type v'
      inst✝⁴ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Limits.HasPullbacks D
      inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
      F : CategoryTheory.Functor J (CategoryTheory.Functor C D)
      c : CategoryTheory.Limits.Cocone F
      hc : ∀ (x : C), CategoryTheory.IsVanKampenColimit (((CategoryTheory.evaluation …
      F' : CategoryTheory.Functor J (CategoryTheory.Functor C D)
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' F
      f : Quiver.Hom c'.pt c.pt
      e : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStru …
      hα : CategoryTheory.NatTrans.Equifibered α
      this : ∀ (x : C), Iff (Nonempty (CategoryTheory.Limits.IsColimit (((CategoryTh …
      hc' : CategoryTheory.Limits.IsColimit c'
      j : J
      ⊢ (k : C) → CategoryTheory.Limits.IsLimit (((CategoryTheory.evaluation C D).ob …
    -/
    refine fun x => (isLimitMapConePullbackConeEquiv _ _).symm ?_
    /-
      case mp.intro
      J : Type v'
      inst✝⁴ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Limits.HasPullbacks D
      inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
      F : CategoryTheory.Functor J (CategoryTheory.Functor C D)
      c : CategoryTheory.Limits.Cocone F
      hc : ∀ (x : C), CategoryTheory.IsVanKampenColimit (((CategoryTheory.evaluation …
      F' : CategoryTheory.Functor J (CategoryTheory.Functor C D)
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' F
      f : Quiver.Hom c'.pt c.pt
      e : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStru …
      hα : CategoryTheory.NatTrans.Equifibered α
      this : ∀ (x : C), Iff (Nonempty (CategoryTheory.Limits.IsColimit (((CategoryTh …
      hc' : CategoryTheory.Limits.IsColimit c'
      j : J
      x : C
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (((Cate …
    -/
    exact ((this x).mp ⟨isColimitOfPreserves _ hc'⟩ _).isLimit
    /-
      🎉 no goals
    -/
  · exact fun H => ⟨evaluationJointlyReflectsColimits _ fun x =>
      ((this x).mpr fun j => (H j).map ((evaluation C D).obj x)).some⟩


theorem IsUniversalColimit.map_reflective
    {Gl : C ⥤ D} {Gr : D ⥤ C} (adj : Gl ⊣ Gr) [Gr.Full] [Gr.Faithful]
    {F : J ⥤ D} {c : Cocone (F ⋙ Gr)}
    (H : IsUniversalColimit c)
    [∀ X (f : X ⟶ Gl.obj c.pt), HasPullback (Gr.map f) (adj.unit.app c.pt)]
    [∀ X (f : X ⟶ Gl.obj c.pt), PreservesLimit (cospan (Gr.map f) (adj.unit.app c.pt)) Gl] :
    IsUniversalColimit (Gl.mapCocone c) := by
  /-
    J : Type v'
    inst✝⁶ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝³ : Gr.Full
    inst✝² : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsUniversalColimit c
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
    ⊢ CategoryTheory.IsUniversalColimit (Gl.mapCocone c)
  -/
  have := adj.rightAdjoint_preservesLimits
  /-
    J : Type v'
    inst✝⁶ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝³ : Gr.Full
    inst✝² : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsUniversalColimit c
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
    this : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.156759, ?u.156758, u_3, …
    ⊢ CategoryTheory.IsUniversalColimit (Gl.mapCocone c)
  -/
  have : PreservesColimitsOfSize.{u', v'} Gl := adj.leftAdjoint_preservesColimits
  /-
    J : Type v'
    inst✝⁶ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝³ : Gr.Full
    inst✝² : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsUniversalColimit c
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
    this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.156759, ?u.156758, u_3 …
    this : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2} Gl
    ⊢ CategoryTheory.IsUniversalColimit (Gl.mapCocone c)
  -/
  intros F' c' α f h hα hc'
  have : HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.unit.app c.pt)) :=
    ⟨⟨_, isLimitPullbackConeMapOfIsLimit _ pullback.condition
      (IsPullback.of_hasPullback _ _).isLimit⟩⟩
  /-
    J : Type v'
    inst✝⁶ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝³ : Gr.Full
    inst✝² : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsUniversalColimit c
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
    this✝¹ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.156759, ?u.156758, u_ …
    this✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2} …
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
    this : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.unit …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c')
  -/
  let α' := α ≫ (Functor.associator _ _ _).hom ≫ whiskerLeft F adj.counit ≫ F.rightUnitor.hom
  /-
    J : Type v'
    inst✝⁶ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝³ : Gr.Full
    inst✝² : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsUniversalColimit c
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
    this✝¹ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.156759, ?u.156758, u_ …
    this✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2} …
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
    this : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.unit …
    α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c')
  -/
  have hα' : NatTrans.Equifibered α' := hα.comp (NatTrans.equifibered_of_isIso _)
  have hadj : ∀ X, Gl.map (adj.unit.app X) = inv (adj.counit.app _) := by
    intro X
    apply IsIso.eq_inv_of_inv_hom_id
    exact adj.left_triangle_components _
  haveI : ∀ X, IsIso (Gl.map (adj.unit.app X)) := by
    simp_rw [hadj]
    infer_instance
  have hα'' : ∀ j, Gl.map (Gr.map <| α'.app j) = adj.counit.app _ ≫ α.app j := by
    intro j
    rw [← cancel_mono (adj.counit.app <| F.obj j)]
    dsimp [α']
    simp only [Category.comp_id, Adjunction.counit_naturality_assoc, Category.id_comp,
      Adjunction.counit_naturality, Category.assoc, Functor.map_comp]
  /-
    J : Type v'
    inst✝⁶ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝³ : Gr.Full
    inst✝² : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsUniversalColimit c
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
    this✝² : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.156759, ?u.156758, u_ …
    this✝¹ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
    this✝ : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.uni …
    α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
    hα' : CategoryTheory.NatTrans.Equifibered α'
    hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
    this : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
    hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c')
  -/
  have hc'' : ∀ j, α.app j ≫ Gl.map (c.ι.app j) = c'.ι.app j ≫ f := NatTrans.congr_app h
  /-
    J : Type v'
    inst✝⁶ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝³ : Gr.Full
    inst✝² : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsUniversalColimit c
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
    this✝² : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.156759, ?u.156758, u_ …
    this✝¹ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
    this✝ : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.uni …
    α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
    hα' : CategoryTheory.NatTrans.Equifibered α'
    hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
    this : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
    hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
    hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c')
  -/
  let β := isoWhiskerLeft F' (asIso adj.counit) ≪≫ F'.rightUnitor
  let c'' : Cocone (F' ⋙ Gr) := by
    refine
    { pt := pullback (Gr.map f) (adj.unit.app _)
      ι := { app := fun j ↦ pullback.lift (Gr.map <| c'.ι.app j) (Gr.map (α'.app j) ≫ c.ι.app j) ?_
             naturality := ?_ } }
    · rw [← Gr.map_comp, ← hc'']
      erw [← adj.unit_naturality]
      rw [Gl.map_comp, hα'']
      dsimp
      simp only [Category.assoc, Functor.map_comp, adj.right_triangle_components_assoc]
    · intros i j g
      dsimp [α']
      ext
      all_goals simp only [Category.comp_id, Category.id_comp, Category.assoc,
        ← Functor.map_comp, pullback.lift_fst, pullback.lift_snd, ← Functor.map_comp_assoc]
      · congr 1
        exact c'.w _
      · rw [α.naturality_assoc]
        dsimp
        rw [adj.counit_naturality, ← Category.assoc, Gr.map_comp_assoc]
        congr 1
        exact c.w _
  let cf : (Cocones.precompose β.hom).obj c' ⟶ Gl.mapCocone c'' := by
    refine { hom := pullback.lift ?_ f ?_ ≫ (PreservesPullback.iso _ _ _).inv, w := ?_ }
    · exact inv <| adj.counit.app c'.pt
    · rw [IsIso.inv_comp_eq, ← adj.counit_naturality_assoc f, ← cancel_mono (adj.counit.app <|
        Gl.obj c.pt), Category.assoc, Category.assoc, adj.left_triangle_components]
      erw [Category.comp_id]
      rfl
    · intro j
      rw [← Category.assoc, Iso.comp_inv_eq]
      ext
      all_goals simp only [c'', PreservesPullback.iso_hom_fst, PreservesPullback.iso_hom_snd,
          pullback.lift_fst, pullback.lift_snd, Category.assoc,
          Functor.mapCocone_ι_app, ← Gl.map_comp]
      · rw [IsIso.comp_inv_eq, adj.counit_naturality]
        dsimp [β]
        rw [Category.comp_id]
      · rw [Gl.map_comp, hα'', Category.assoc, hc'']
        dsimp [β]
        rw [Category.comp_id, Category.assoc]
  have :
      cf.hom ≫ (PreservesPullback.iso _ _ _).hom ≫ pullback.fst _ _ ≫ adj.counit.app _ = 𝟙 _ := by
    simp only [cf, IsIso.inv_hom_id, Iso.inv_hom_id_assoc, Category.assoc,
      pullback.lift_fst_assoc]
  have : IsIso cf := by
    apply @Cocones.cocone_iso_of_hom_iso (i := ?_)
    rw [← IsIso.eq_comp_inv] at this
    rw [this]
    infer_instance
  /-
    J : Type v'
    inst✝⁶ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝³ : Gr.Full
    inst✝² : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsUniversalColimit c
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
    this✝⁴ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.156759, ?u.156758, u_ …
    this✝³ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
    this✝² : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.un …
    α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
    hα' : CategoryTheory.NatTrans.Equifibered α'
    hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
    this✝¹ : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
    hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
    hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
    β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
    c'' : CategoryTheory.Limits.Cocone (F'.comp Gr) := { pt := CategoryTheory.Limi …
    cf : Quiver.Hom ((CategoryTheory.Limits.Cocones.precompose β.hom).obj c') (Gl. …
    this✝ : Eq (CategoryTheory.CategoryStruct.comp cf.hom (CategoryTheory.Category …
    this : CategoryTheory.IsIso cf
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c')
  -/
  have ⟨Hc''⟩ := H c'' (whiskerRight α' Gr) (pullback.snd _ _) ?_ (hα'.whiskerRight Gr) ?_
  · exact ⟨IsColimit.precomposeHomEquiv β c' <|
      (isColimitOfPreserves Gl Hc'').ofIsoColimit (asIso cf).symm⟩
    /-
      case refine_1
      J : Type v'
      inst✝⁶ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝³ : Gr.Full
      inst✝² : Gr.Faithful
      F : CategoryTheory.Functor J D
      c : CategoryTheory.Limits.Cocone (F.comp Gr)
      H : CategoryTheory.IsUniversalColimit c
      inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
      inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
      this✝⁴ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
      this✝³ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
      F' : CategoryTheory.Functor J D
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' ((F.comp Gr).comp Gl)
      f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
      h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
      hα : CategoryTheory.NatTrans.Equifibered α
      hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
      this✝² : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.un …
      α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
      hα' : CategoryTheory.NatTrans.Equifibered α'
      hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
      this✝¹ : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
      hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
      hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
      β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
      c'' : CategoryTheory.Limits.Cocone (F'.comp Gr) := { pt := CategoryTheory.Limi …
      cf : Quiver.Hom ((CategoryTheory.Limits.Cocones.precompose β.hom).obj c') (Gl. …
      this✝ : Eq (CategoryTheory.CategoryStruct.comp cf.hom (CategoryTheory.Category …
      this : CategoryTheory.IsIso cf
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight α' Gr) c …
    -/
  · ext j
    /-
      case refine_1.w.h
      J : Type v'
      inst✝⁶ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝³ : Gr.Full
      inst✝² : Gr.Faithful
      F : CategoryTheory.Functor J D
      c : CategoryTheory.Limits.Cocone (F.comp Gr)
      H : CategoryTheory.IsUniversalColimit c
      inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
      inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
      this✝⁴ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
      this✝³ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
      F' : CategoryTheory.Functor J D
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' ((F.comp Gr).comp Gl)
      f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
      h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
      hα : CategoryTheory.NatTrans.Equifibered α
      hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
      this✝² : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.un …
      α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
      hα' : CategoryTheory.NatTrans.Equifibered α'
      hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
      this✝¹ : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
      hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
      hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
      β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
      c'' : CategoryTheory.Limits.Cocone (F'.comp Gr) := { pt := CategoryTheory.Limi …
      cf : Quiver.Hom ((CategoryTheory.Limits.Cocones.precompose β.hom).obj c') (Gl. …
      this✝ : Eq (CategoryTheory.CategoryStruct.comp cf.hom (CategoryTheory.Category …
      this : CategoryTheory.IsIso cf
      j : J
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight α' Gr)  …
    -/
    dsimp [c'']
    simp only [Category.comp_id, Category.id_comp, Category.assoc,
      Functor.map_comp, pullback.lift_snd]
    /-
      case refine_2
      J : Type v'
      inst✝⁶ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝³ : Gr.Full
      inst✝² : Gr.Faithful
      F : CategoryTheory.Functor J D
      c : CategoryTheory.Limits.Cocone (F.comp Gr)
      H : CategoryTheory.IsUniversalColimit c
      inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
      inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
      this✝⁴ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
      this✝³ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
      F' : CategoryTheory.Functor J D
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' ((F.comp Gr).comp Gl)
      f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
      h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
      hα : CategoryTheory.NatTrans.Equifibered α
      hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
      this✝² : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.un …
      α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
      hα' : CategoryTheory.NatTrans.Equifibered α'
      hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
      this✝¹ : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
      hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
      hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
      β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
      c'' : CategoryTheory.Limits.Cocone (F'.comp Gr) := { pt := CategoryTheory.Limi …
      cf : Quiver.Hom ((CategoryTheory.Limits.Cocones.precompose β.hom).obj c') (Gl. …
      this✝ : Eq (CategoryTheory.CategoryStruct.comp cf.hom (CategoryTheory.Category …
      this : CategoryTheory.IsIso cf
      ⊢ ∀ (j : J), CategoryTheory.IsPullback (c''.ι.app j) ((CategoryTheory.whiskerR …
    -/
  · intro j
    /-
      case refine_2
      J : Type v'
      inst✝⁶ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝³ : Gr.Full
      inst✝² : Gr.Faithful
      F : CategoryTheory.Functor J D
      c : CategoryTheory.Limits.Cocone (F.comp Gr)
      H : CategoryTheory.IsUniversalColimit c
      inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
      inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
      this✝⁴ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
      this✝³ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
      F' : CategoryTheory.Functor J D
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' ((F.comp Gr).comp Gl)
      f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
      h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
      hα : CategoryTheory.NatTrans.Equifibered α
      hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
      this✝² : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.un …
      α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
      hα' : CategoryTheory.NatTrans.Equifibered α'
      hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
      this✝¹ : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
      hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
      hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
      β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
      c'' : CategoryTheory.Limits.Cocone (F'.comp Gr) := { pt := CategoryTheory.Limi …
      cf : Quiver.Hom ((CategoryTheory.Limits.Cocones.precompose β.hom).obj c') (Gl. …
      this✝ : Eq (CategoryTheory.CategoryStruct.comp cf.hom (CategoryTheory.Category …
      this : CategoryTheory.IsIso cf
      j : J
      ⊢ CategoryTheory.IsPullback (c''.ι.app j) ((CategoryTheory.whiskerRight α' Gr) …
    -/
    apply IsPullback.of_right _ _ (IsPullback.of_hasPullback _ _)
      /-
        J : Type v'
        inst✝⁶ : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
        Gl : CategoryTheory.Functor C D
        Gr : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction Gl Gr
        inst✝³ : Gr.Full
        inst✝² : Gr.Faithful
        F : CategoryTheory.Functor J D
        c : CategoryTheory.Limits.Cocone (F.comp Gr)
        H : CategoryTheory.IsUniversalColimit c
        inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
        inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
        this✝⁴ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
        this✝³ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
        F' : CategoryTheory.Functor J D
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' ((F.comp Gr).comp Gl)
        f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
        h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
        hα : CategoryTheory.NatTrans.Equifibered α
        hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
        this✝² : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.un …
        α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
        hα' : CategoryTheory.NatTrans.Equifibered α'
        hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
        this✝¹ : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
        hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
        hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
        β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
        c'' : CategoryTheory.Limits.Cocone (F'.comp Gr) := { pt := CategoryTheory.Limi …
        cf : Quiver.Hom ((CategoryTheory.Limits.Cocones.precompose β.hom).obj c') (Gl. …
        this✝ : Eq (CategoryTheory.CategoryStruct.comp cf.hom (CategoryTheory.Category …
        this : CategoryTheory.IsIso cf
        j : J
        ⊢ CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.comp (c''.ι.app j)  …
      -/
    · dsimp [α', c'']
      simp only [Category.comp_id, Category.id_comp, Category.assoc, Functor.map_comp,
        pullback.lift_fst]
      /-
        J : Type v'
        inst✝⁶ : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
        Gl : CategoryTheory.Functor C D
        Gr : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction Gl Gr
        inst✝³ : Gr.Full
        inst✝² : Gr.Faithful
        F : CategoryTheory.Functor J D
        c : CategoryTheory.Limits.Cocone (F.comp Gr)
        H : CategoryTheory.IsUniversalColimit c
        inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
        inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
        this✝⁴ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
        this✝³ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
        F' : CategoryTheory.Functor J D
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' ((F.comp Gr).comp Gl)
        f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
        h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
        hα : CategoryTheory.NatTrans.Equifibered α
        hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
        this✝² : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.un …
        α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
        hα' : CategoryTheory.NatTrans.Equifibered α'
        hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
        this✝¹ : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
        hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
        hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
        β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
        c'' : CategoryTheory.Limits.Cocone (F'.comp Gr) := { pt := CategoryTheory.Limi …
        cf : Quiver.Hom ((CategoryTheory.Limits.Cocones.precompose β.hom).obj c') (Gl. …
        this✝ : Eq (CategoryTheory.CategoryStruct.comp cf.hom (CategoryTheory.Category …
        this : CategoryTheory.IsIso cf
        j : J
        ⊢ CategoryTheory.IsPullback (Gr.map (c'.ι.app j)) (CategoryTheory.CategoryStru …
      -/
      rw [← Category.comp_id (Gr.map f)]
      /-
        J : Type v'
        inst✝⁶ : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
        Gl : CategoryTheory.Functor C D
        Gr : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction Gl Gr
        inst✝³ : Gr.Full
        inst✝² : Gr.Faithful
        F : CategoryTheory.Functor J D
        c : CategoryTheory.Limits.Cocone (F.comp Gr)
        H : CategoryTheory.IsUniversalColimit c
        inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
        inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
        this✝⁴ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
        this✝³ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
        F' : CategoryTheory.Functor J D
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' ((F.comp Gr).comp Gl)
        f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
        h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
        hα : CategoryTheory.NatTrans.Equifibered α
        hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
        this✝² : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.un …
        α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
        hα' : CategoryTheory.NatTrans.Equifibered α'
        hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
        this✝¹ : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
        hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
        hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
        β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
        c'' : CategoryTheory.Limits.Cocone (F'.comp Gr) := { pt := CategoryTheory.Limi …
        cf : Quiver.Hom ((CategoryTheory.Limits.Cocones.precompose β.hom).obj c') (Gl. …
        this✝ : Eq (CategoryTheory.CategoryStruct.comp cf.hom (CategoryTheory.Category …
        this : CategoryTheory.IsIso cf
        j : J
        ⊢ CategoryTheory.IsPullback (Gr.map (c'.ι.app j)) (CategoryTheory.CategoryStru …
      -/
      refine ((hc' j).map Gr).paste_vert (IsPullback.of_vert_isIso ⟨?_⟩)
      rw [← adj.unit_naturality, Category.comp_id, ← Category.assoc,
        ← Category.id_comp (Gr.map ((Gl.mapCocone c).ι.app j))]
      /-
        J : Type v'
        inst✝⁶ : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
        Gl : CategoryTheory.Functor C D
        Gr : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction Gl Gr
        inst✝³ : Gr.Full
        inst✝² : Gr.Faithful
        F : CategoryTheory.Functor J D
        c : CategoryTheory.Limits.Cocone (F.comp Gr)
        H : CategoryTheory.IsUniversalColimit c
        inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
        inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
        this✝⁴ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
        this✝³ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
        F' : CategoryTheory.Functor J D
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' ((F.comp Gr).comp Gl)
        f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
        h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
        hα : CategoryTheory.NatTrans.Equifibered α
        hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
        this✝² : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.un …
        α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
        hα' : CategoryTheory.NatTrans.Equifibered α'
        hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
        this✝¹ : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
        hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
        hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
        β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
        c'' : CategoryTheory.Limits.Cocone (F'.comp Gr) := { pt := CategoryTheory.Limi …
        cf : Quiver.Hom ((CategoryTheory.Limits.Cocones.precompose β.hom).obj c') (Gl. …
        this✝ : Eq (CategoryTheory.CategoryStruct.comp cf.hom (CategoryTheory.Category …
        this : CategoryTheory.IsIso cf
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Gr …
      -/
      congr 1
      /-
        case e_a
        J : Type v'
        inst✝⁶ : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
        Gl : CategoryTheory.Functor C D
        Gr : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction Gl Gr
        inst✝³ : Gr.Full
        inst✝² : Gr.Faithful
        F : CategoryTheory.Functor J D
        c : CategoryTheory.Limits.Cocone (F.comp Gr)
        H : CategoryTheory.IsUniversalColimit c
        inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
        inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
        this✝⁴ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
        this✝³ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
        F' : CategoryTheory.Functor J D
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' ((F.comp Gr).comp Gl)
        f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
        h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
        hα : CategoryTheory.NatTrans.Equifibered α
        hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
        this✝² : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.un …
        α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
        hα' : CategoryTheory.NatTrans.Equifibered α'
        hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
        this✝¹ : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
        hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
        hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
        β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
        c'' : CategoryTheory.Limits.Cocone (F'.comp Gr) := { pt := CategoryTheory.Limi …
        cf : Quiver.Hom ((CategoryTheory.Limits.Cocones.precompose β.hom).obj c') (Gl. …
        this✝ : Eq (CategoryTheory.CategoryStruct.comp cf.hom (CategoryTheory.Category …
        this : CategoryTheory.IsIso cf
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.id (Gr.obj (((F.comp Gr).comp Gl).obj j))) …
      -/
      rw [← cancel_mono (Gr.map (adj.counit.app (F.obj j)))]
      /-
        case e_a
        J : Type v'
        inst✝⁶ : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
        Gl : CategoryTheory.Functor C D
        Gr : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction Gl Gr
        inst✝³ : Gr.Full
        inst✝² : Gr.Faithful
        F : CategoryTheory.Functor J D
        c : CategoryTheory.Limits.Cocone (F.comp Gr)
        H : CategoryTheory.IsUniversalColimit c
        inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
        inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
        this✝⁴ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
        this✝³ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
        F' : CategoryTheory.Functor J D
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' ((F.comp Gr).comp Gl)
        f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
        h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
        hα : CategoryTheory.NatTrans.Equifibered α
        hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
        this✝² : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.un …
        α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
        hα' : CategoryTheory.NatTrans.Equifibered α'
        hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
        this✝¹ : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
        hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
        hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
        β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
        c'' : CategoryTheory.Limits.Cocone (F'.comp Gr) := { pt := CategoryTheory.Limi …
        cf : Quiver.Hom ((CategoryTheory.Limits.Cocones.precompose β.hom).obj c') (Gl. …
        this✝ : Eq (CategoryTheory.CategoryStruct.comp cf.hom (CategoryTheory.Category …
        this : CategoryTheory.IsIso cf
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Gr …
      -/
      dsimp
      simp only [Category.comp_id, Adjunction.right_triangle_components, Category.id_comp,
        Category.assoc]
      /-
        J : Type v'
        inst✝⁶ : CategoryTheory.Category.{u', v'} J
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
        Gl : CategoryTheory.Functor C D
        Gr : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction Gl Gr
        inst✝³ : Gr.Full
        inst✝² : Gr.Faithful
        F : CategoryTheory.Functor J D
        c : CategoryTheory.Limits.Cocone (F.comp Gr)
        H : CategoryTheory.IsUniversalColimit c
        inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
        inst✝ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pres …
        this✝⁴ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
        this✝³ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
        F' : CategoryTheory.Functor J D
        c' : CategoryTheory.Limits.Cocone F'
        α : Quiver.Hom F' ((F.comp Gr).comp Gl)
        f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
        h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
        hα : CategoryTheory.NatTrans.Equifibered α
        hc' : ∀ (j : J), CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCo …
        this✝² : CategoryTheory.Limits.HasPullback (Gl.map (Gr.map f)) (Gl.map (adj.un …
        α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
        hα' : CategoryTheory.NatTrans.Equifibered α'
        hadj : ∀ (X : C), Eq (Gl.map (adj.unit.app X)) (CategoryTheory.inv (adj.counit …
        this✝¹ : ∀ (X : C), CategoryTheory.IsIso (Gl.map (adj.unit.app X))
        hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
        hc'' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (α.app j) (Gl.map (c. …
        β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
        c'' : CategoryTheory.Limits.Cocone (F'.comp Gr) := { pt := CategoryTheory.Limi …
        cf : Quiver.Hom ((CategoryTheory.Limits.Cocones.precompose β.hom).obj c') (Gl. …
        this✝ : Eq (CategoryTheory.CategoryStruct.comp cf.hom (CategoryTheory.Category …
        this : CategoryTheory.IsIso cf
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c''.ι.app j) (CategoryTheory.Limits. …
      -/
    · dsimp [c'']
      simp only [Category.comp_id, Category.id_comp, Category.assoc, Functor.map_comp,
        pullback.lift_snd]


theorem IsVanKampenColimit.map_reflective [HasColimitsOfShape J C]
    {Gl : C ⥤ D} {Gr : D ⥤ C} (adj : Gl ⊣ Gr) [Gr.Full] [Gr.Faithful]
    {F : J ⥤ D} {c : Cocone (F ⋙ Gr)} (H : IsVanKampenColimit c)
    [∀ X (f : X ⟶ Gl.obj c.pt), HasPullback (Gr.map f) (adj.unit.app c.pt)]
    [∀ X (f : X ⟶ Gl.obj c.pt), PreservesLimit (cospan (Gr.map f) (adj.unit.app c.pt)) Gl]
    [∀ X i (f : X ⟶ c.pt), PreservesLimit (cospan f (c.ι.app i)) Gl] :
    IsVanKampenColimit (Gl.mapCocone c) := by
  /-
    J : Type v'
    inst✝⁸ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsVanKampenColimit c
    inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
    inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
    ⊢ CategoryTheory.IsVanKampenColimit (Gl.mapCocone c)
  -/
  have := adj.rightAdjoint_preservesLimits
  /-
    J : Type v'
    inst✝⁸ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsVanKampenColimit c
    inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
    inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
    this : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.206679, ?u.206678, u_3, …
    ⊢ CategoryTheory.IsVanKampenColimit (Gl.mapCocone c)
  -/
  have : PreservesColimitsOfSize.{u', v'} Gl := adj.leftAdjoint_preservesColimits
  /-
    J : Type v'
    inst✝⁸ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsVanKampenColimit c
    inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
    inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
    this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.206679, ?u.206678, u_3 …
    this : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2} Gl
    ⊢ CategoryTheory.IsVanKampenColimit (Gl.mapCocone c)
  -/
  intro F' c' α f h hα
  /-
    J : Type v'
    inst✝⁸ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsVanKampenColimit c
    inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
    inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
    this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.206679, ?u.206678, u_3 …
    this : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2} Gl
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (∀ (j : J), CategoryTheo …
  -/
  refine ⟨?_, H.isUniversal.map_reflective adj c' α f h hα⟩
  /-
    J : Type v'
    inst✝⁸ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsVanKampenColimit c
    inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
    inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
    this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.206679, ?u.206678, u_3 …
    this : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2} Gl
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c') → ∀ (j : J), CategoryTheory.Is …
  -/
  intro ⟨hc'⟩ j
  /-
    J : Type v'
    inst✝⁸ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsVanKampenColimit c
    inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
    inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
    this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.206679, ?u.206678, u_3 …
    this : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2} Gl
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    hc' : CategoryTheory.Limits.IsColimit c'
    j : J
    ⊢ CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCocone c).ι.app j)
  -/
  let α' := α ≫ (Functor.associator _ _ _).hom ≫ whiskerLeft F adj.counit ≫ F.rightUnitor.hom
  /-
    J : Type v'
    inst✝⁸ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsVanKampenColimit c
    inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
    inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
    this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.206679, ?u.206678, u_3 …
    this : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2} Gl
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    hc' : CategoryTheory.Limits.IsColimit c'
    j : J
    α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
    ⊢ CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCocone c).ι.app j)
  -/
  have hα' : NatTrans.Equifibered α' := hα.comp (NatTrans.equifibered_of_isIso _)
  have hα'' : ∀ j, Gl.map (Gr.map <| α'.app j) = adj.counit.app _ ≫ α.app j := by
    intro j
    rw [← cancel_mono (adj.counit.app <| F.obj j)]
    dsimp [α']
    simp only [Category.comp_id, Adjunction.counit_naturality_assoc, Category.id_comp,
      Adjunction.counit_naturality, Category.assoc, Functor.map_comp]
  /-
    J : Type v'
    inst✝⁸ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsVanKampenColimit c
    inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
    inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
    this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.206679, ?u.206678, u_3 …
    this : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2} Gl
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    hc' : CategoryTheory.Limits.IsColimit c'
    j : J
    α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
    hα' : CategoryTheory.NatTrans.Equifibered α'
    hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
    ⊢ CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCocone c).ι.app j)
  -/
  let β := isoWhiskerLeft F' (asIso adj.counit) ≪≫ F'.rightUnitor
  /-
    J : Type v'
    inst✝⁸ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsVanKampenColimit c
    inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
    inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
    this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.206679, ?u.206678, u_3 …
    this : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2} Gl
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    hc' : CategoryTheory.Limits.IsColimit c'
    j : J
    α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
    hα' : CategoryTheory.NatTrans.Equifibered α'
    hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
    β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
    ⊢ CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCocone c).ι.app j)
  -/
  let hl := (IsColimit.precomposeHomEquiv β c').symm hc'
  /-
    J : Type v'
    inst✝⁸ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsVanKampenColimit c
    inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
    inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
    this✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.206679, ?u.206678, u_3 …
    this : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2} Gl
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    hc' : CategoryTheory.Limits.IsColimit c'
    j : J
    α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
    hα' : CategoryTheory.NatTrans.Equifibered α'
    hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
    β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
    hl : CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompos …
    ⊢ CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCocone c).ι.app j)
  -/
  let hr := isColimitOfPreserves Gl (colimit.isColimit <| F' ⋙ Gr)
  have : α.app j = β.inv.app _ ≫ Gl.map (Gr.map <| α'.app j) := by
    rw [hα'']
    simp [β]
  /-
    J : Type v'
    inst✝⁸ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsVanKampenColimit c
    inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
    inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
    this✝¹ : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.206679, ?u.206678, u_ …
    this✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2} …
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    hc' : CategoryTheory.Limits.IsColimit c'
    j : J
    α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
    hα' : CategoryTheory.NatTrans.Equifibered α'
    hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
    β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
    hl : CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompos …
    hr : CategoryTheory.Limits.IsColimit (Gl.mapCocone (CategoryTheory.Limits.coli …
    this : Eq (α.app j) (CategoryTheory.CategoryStruct.comp (β.inv.app j) (Gl.map  …
    ⊢ CategoryTheory.IsPullback (c'.ι.app j) (α.app j) f ((Gl.mapCocone c).ι.app j)
  -/
  rw [this]
  have : f = (hl.coconePointUniqueUpToIso hr).hom ≫
    Gl.map (colimit.desc _ ⟨_, whiskerRight α' Gr ≫ c.2⟩) := by
    symm
    convert @IsColimit.coconePointUniqueUpToIso_hom_desc _ _ _ _ ((F' ⋙ Gr) ⋙ Gl)
      (Gl.mapCocone ⟨_, (whiskerRight α' Gr ≫ c.2 : _)⟩) _ _ hl hr using 2
    · apply hr.hom_ext
      intro j
      rw [hr.fac, Functor.mapCocone_ι_app, ← Gl.map_comp, colimit.cocone_ι, colimit.ι_desc]
      rfl
    · clear_value α'
      apply hl.hom_ext
      intro j
      rw [hl.fac]
      dsimp [β]
      simp only [Category.comp_id, hα'', Category.assoc, Gl.map_comp]
      congr 1
      exact (NatTrans.congr_app h j).symm
  /-
    J : Type v'
    inst✝⁸ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
    Gl : CategoryTheory.Functor C D
    Gr : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction Gl Gr
    inst✝⁴ : Gr.Full
    inst✝³ : Gr.Faithful
    F : CategoryTheory.Functor J D
    c : CategoryTheory.Limits.Cocone (F.comp Gr)
    H : CategoryTheory.IsVanKampenColimit c
    inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
    inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
    inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
    this✝² : CategoryTheory.Limits.PreservesLimitsOfSize.{?u.206679, ?u.206678, u_ …
    this✝¹ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
    F' : CategoryTheory.Functor J D
    c' : CategoryTheory.Limits.Cocone F'
    α : Quiver.Hom F' ((F.comp Gr).comp Gl)
    f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
    h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
    hα : CategoryTheory.NatTrans.Equifibered α
    hc' : CategoryTheory.Limits.IsColimit c'
    j : J
    α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
    hα' : CategoryTheory.NatTrans.Equifibered α'
    hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
    β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
    hl : CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompos …
    hr : CategoryTheory.Limits.IsColimit (Gl.mapCocone (CategoryTheory.Limits.coli …
    this✝ : Eq (α.app j) (CategoryTheory.CategoryStruct.comp (β.inv.app j) (Gl.map …
    this : Eq f (CategoryTheory.CategoryStruct.comp (hl.coconePointUniqueUpToIso h …
    ⊢ CategoryTheory.IsPullback (c'.ι.app j) (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [this]
  have := ((H (colimit.cocone <| F' ⋙ Gr) (whiskerRight α' Gr)
    (colimit.desc _ ⟨_, whiskerRight α' Gr ≫ c.2⟩) ?_ (hα'.whiskerRight Gr)).mp
    ⟨(getColimitCocone <| F' ⋙ Gr).2⟩ j).map Gl
    /-
      case refine_2
      J : Type v'
      inst✝⁸ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝⁴ : Gr.Full
      inst✝³ : Gr.Faithful
      F : CategoryTheory.Functor J D
      c : CategoryTheory.Limits.Cocone (F.comp Gr)
      H : CategoryTheory.IsVanKampenColimit c
      inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
      inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
      inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
      this✝³ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
      this✝² : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
      F' : CategoryTheory.Functor J D
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' ((F.comp Gr).comp Gl)
      f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
      h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
      hα : CategoryTheory.NatTrans.Equifibered α
      hc' : CategoryTheory.Limits.IsColimit c'
      j : J
      α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
      hα' : CategoryTheory.NatTrans.Equifibered α'
      hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
      β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
      hl : CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompos …
      hr : CategoryTheory.Limits.IsColimit (Gl.mapCocone (CategoryTheory.Limits.coli …
      this✝¹ : Eq (α.app j) (CategoryTheory.CategoryStruct.comp (β.inv.app j) (Gl.ma …
      this✝ : Eq f (CategoryTheory.CategoryStruct.comp (hl.coconePointUniqueUpToIso  …
      this : CategoryTheory.IsPullback (Gl.map ((CategoryTheory.Limits.colimit.cocon …
      ⊢ CategoryTheory.IsPullback (c'.ι.app j) (CategoryTheory.CategoryStruct.comp ( …
    -/
  · convert IsPullback.paste_vert _ this
    /-
      case refine_2.convert_6
      J : Type v'
      inst✝⁸ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝⁴ : Gr.Full
      inst✝³ : Gr.Faithful
      F : CategoryTheory.Functor J D
      c : CategoryTheory.Limits.Cocone (F.comp Gr)
      H : CategoryTheory.IsVanKampenColimit c
      inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
      inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
      inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
      this✝³ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
      this✝² : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
      F' : CategoryTheory.Functor J D
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' ((F.comp Gr).comp Gl)
      f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
      h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
      hα : CategoryTheory.NatTrans.Equifibered α
      hc' : CategoryTheory.Limits.IsColimit c'
      j : J
      α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
      hα' : CategoryTheory.NatTrans.Equifibered α'
      hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
      β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
      hl : CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompos …
      hr : CategoryTheory.Limits.IsColimit (Gl.mapCocone (CategoryTheory.Limits.coli …
      this✝¹ : Eq (α.app j) (CategoryTheory.CategoryStruct.comp (β.inv.app j) (Gl.ma …
      this✝ : Eq f (CategoryTheory.CategoryStruct.comp (hl.coconePointUniqueUpToIso  …
      this : CategoryTheory.IsPullback (Gl.map ((CategoryTheory.Limits.colimit.cocon …
      ⊢ CategoryTheory.IsPullback (c'.ι.app j) (β.inv.app j) (hl.coconePointUniqueUp …
    -/
    refine IsPullback.of_vert_isIso ⟨?_⟩
    /-
      case refine_2.convert_6
      J : Type v'
      inst✝⁸ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝⁴ : Gr.Full
      inst✝³ : Gr.Faithful
      F : CategoryTheory.Functor J D
      c : CategoryTheory.Limits.Cocone (F.comp Gr)
      H : CategoryTheory.IsVanKampenColimit c
      inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
      inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
      inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
      this✝³ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
      this✝² : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
      F' : CategoryTheory.Functor J D
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' ((F.comp Gr).comp Gl)
      f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
      h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
      hα : CategoryTheory.NatTrans.Equifibered α
      hc' : CategoryTheory.Limits.IsColimit c'
      j : J
      α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
      hα' : CategoryTheory.NatTrans.Equifibered α'
      hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
      β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
      hl : CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompos …
      hr : CategoryTheory.Limits.IsColimit (Gl.mapCocone (CategoryTheory.Limits.coli …
      this✝¹ : Eq (α.app j) (CategoryTheory.CategoryStruct.comp (β.inv.app j) (Gl.ma …
      this✝ : Eq f (CategoryTheory.CategoryStruct.comp (hl.coconePointUniqueUpToIso  …
      this : CategoryTheory.IsPullback (Gl.map ((CategoryTheory.Limits.colimit.cocon …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c'.ι.app j) (hl.coconePointUniqueUpT …
    -/
    rw [← IsIso.inv_comp_eq, ← Category.assoc, NatIso.inv_inv_app]
    /-
      case refine_2.convert_6
      J : Type v'
      inst✝⁸ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝⁴ : Gr.Full
      inst✝³ : Gr.Faithful
      F : CategoryTheory.Functor J D
      c : CategoryTheory.Limits.Cocone (F.comp Gr)
      H : CategoryTheory.IsVanKampenColimit c
      inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
      inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
      inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
      this✝³ : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
      this✝² : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
      F' : CategoryTheory.Functor J D
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' ((F.comp Gr).comp Gl)
      f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
      h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
      hα : CategoryTheory.NatTrans.Equifibered α
      hc' : CategoryTheory.Limits.IsColimit c'
      j : J
      α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
      hα' : CategoryTheory.NatTrans.Equifibered α'
      hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
      β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
      hl : CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompos …
      hr : CategoryTheory.Limits.IsColimit (Gl.mapCocone (CategoryTheory.Limits.coli …
      this✝¹ : Eq (α.app j) (CategoryTheory.CategoryStruct.comp (β.inv.app j) (Gl.ma …
      this✝ : Eq f (CategoryTheory.CategoryStruct.comp (hl.coconePointUniqueUpToIso  …
      this : CategoryTheory.IsPullback (Gl.map ((CategoryTheory.Limits.colimit.cocon …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    exact IsColimit.comp_coconePointUniqueUpToIso_hom hl hr _
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      J : Type v'
      inst✝⁸ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝⁴ : Gr.Full
      inst✝³ : Gr.Faithful
      F : CategoryTheory.Functor J D
      c : CategoryTheory.Limits.Cocone (F.comp Gr)
      H : CategoryTheory.IsVanKampenColimit c
      inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
      inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
      inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
      this✝² : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
      this✝¹ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
      F' : CategoryTheory.Functor J D
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' ((F.comp Gr).comp Gl)
      f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
      h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
      hα : CategoryTheory.NatTrans.Equifibered α
      hc' : CategoryTheory.Limits.IsColimit c'
      j : J
      α' : Quiver.Hom F' F := CategoryTheory.CategoryStruct.comp α (CategoryTheory.C …
      hα' : CategoryTheory.NatTrans.Equifibered α'
      hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
      β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
      hl : CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompos …
      hr : CategoryTheory.Limits.IsColimit (Gl.mapCocone (CategoryTheory.Limits.coli …
      this✝ : Eq (α.app j) (CategoryTheory.CategoryStruct.comp (β.inv.app j) (Gl.map …
      this : Eq f (CategoryTheory.CategoryStruct.comp (hl.coconePointUniqueUpToIso h …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight α' Gr) c …
    -/
  · clear_value α'
    /-
      case refine_1
      J : Type v'
      inst✝⁸ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝⁴ : Gr.Full
      inst✝³ : Gr.Faithful
      F : CategoryTheory.Functor J D
      c : CategoryTheory.Limits.Cocone (F.comp Gr)
      H : CategoryTheory.IsVanKampenColimit c
      inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
      inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
      inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
      this✝² : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
      this✝¹ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
      F' : CategoryTheory.Functor J D
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' ((F.comp Gr).comp Gl)
      f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
      h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
      hα : CategoryTheory.NatTrans.Equifibered α
      hc' : CategoryTheory.Limits.IsColimit c'
      j : J
      β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
      hl : CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompos …
      hr : CategoryTheory.Limits.IsColimit (Gl.mapCocone (CategoryTheory.Limits.coli …
      α' : Quiver.Hom F' F
      hα' : CategoryTheory.NatTrans.Equifibered α'
      hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
      this✝ : Eq (α.app j) (CategoryTheory.CategoryStruct.comp (β.inv.app j) (Gl.map …
      this : Eq f (CategoryTheory.CategoryStruct.comp (hl.coconePointUniqueUpToIso h …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight α' Gr) c …
    -/
    ext j
    /-
      case refine_1.w.h
      J : Type v'
      inst✝⁸ : CategoryTheory.Category.{u', v'} J
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape J C
      Gl : CategoryTheory.Functor C D
      Gr : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction Gl Gr
      inst✝⁴ : Gr.Full
      inst✝³ : Gr.Faithful
      F : CategoryTheory.Functor J D
      c : CategoryTheory.Limits.Cocone (F.comp Gr)
      H : CategoryTheory.IsVanKampenColimit c
      inst✝² : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Has …
      inst✝¹ : ∀ (X : D) (f : Quiver.Hom X (Gl.obj c.pt)), CategoryTheory.Limits.Pre …
      inst✝ : ∀ (X : C) (i : J) (f : Quiver.Hom X c.pt), CategoryTheory.Limits.Prese …
      this✝² : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, u_3, v, u_2, u} Gr
      this✝¹ : CategoryTheory.Limits.PreservesColimitsOfSize.{u', v', v, u_3, u, u_2 …
      F' : CategoryTheory.Functor J D
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' ((F.comp Gr).comp Gl)
      f : Quiver.Hom c'.pt (Gl.mapCocone c).pt
      h : Eq (CategoryTheory.CategoryStruct.comp α (Gl.mapCocone c).ι) (CategoryTheo …
      hα : CategoryTheory.NatTrans.Equifibered α
      hc' : CategoryTheory.Limits.IsColimit c'
      j✝ : J
      β : CategoryTheory.Iso (F'.comp (Gr.comp Gl)) F' := (CategoryTheory.isoWhisker …
      hl : CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompos …
      hr : CategoryTheory.Limits.IsColimit (Gl.mapCocone (CategoryTheory.Limits.coli …
      α' : Quiver.Hom F' F
      hα' : CategoryTheory.NatTrans.Equifibered α'
      hα'' : ∀ (j : J), Eq (Gl.map (Gr.map (α'.app j))) (CategoryTheory.CategoryStru …
      this✝ : Eq (α.app j✝) (CategoryTheory.CategoryStruct.comp (β.inv.app j✝) (Gl.m …
      this : Eq f (CategoryTheory.CategoryStruct.comp (hl.coconePointUniqueUpToIso h …
      j : J
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight α' Gr)  …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem hasStrictInitial_of_isUniversal [HasInitial C]
    (H : IsUniversalColimit (BinaryCofan.mk (𝟙 (⊥_ C)) (𝟙 (⊥_ C)))) : HasStrictInitialObjects C :=
  hasStrictInitialObjects_of_initial_is_strict
    (by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasInitial C
        H : CategoryTheory.IsUniversalColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
        ⊢ ∀ (A : C) (f : Quiver.Hom A (CategoryTheory.Limits.initial C)), CategoryTheo …
      -/
      intro A f
      suffices IsColimit (BinaryCofan.mk (𝟙 A) (𝟙 A)) by
        obtain ⟨l, h₁, h₂⟩ := Limits.BinaryCofan.IsColimit.desc' this (f ≫ initial.to A) (𝟙 A)
        rcases(Category.id_comp _).symm.trans h₂ with rfl
        exact ⟨⟨_, ((Category.id_comp _).symm.trans h₁).symm, initialIsInitial.hom_ext _ _⟩⟩
      refine (H (BinaryCofan.mk (𝟙 _) (𝟙 _)) (mapPair f f) f (by ext ⟨⟨⟩⟩ <;> dsimp <;> simp)
        (mapPair_equifibered _) ?_).some
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasInitial C
        H : CategoryTheory.IsUniversalColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
        A : C
        f : Quiver.Hom A (CategoryTheory.Limits.initial C)
        ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), CategoryT …
      -/
      rintro ⟨⟨⟩⟩ <;> dsimp <;>
        /-
          case mk.left
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasInitial C
          H : CategoryTheory.IsUniversalColimit (CategoryTheory.Limits.BinaryCofan.mk (C …
          A : C
          f : Quiver.Hom A (CategoryTheory.Limits.initial C)
          ⊢ CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.id A) f f (Category …
        -/
        /-
          🎉 no goals
        -/
        exact IsPullback.of_horiz_isIso ⟨(Category.id_comp _).trans (Category.comp_id _).symm⟩)
        /-
          🎉 no goals
        -/


theorem isVanKampenColimit_of_isEmpty [HasStrictInitialObjects C] [IsEmpty J] {F : J ⥤ C}
    (c : Cocone F) (hc : IsColimit c) : IsVanKampenColimit c := by
  have : IsInitial c.pt := by
    have := (IsColimit.precomposeInvEquiv (Functor.uniqueFromEmpty _) _).symm
      (hc.whiskerEquivalence (equivalenceOfIsEmpty (Discrete PEmpty.{1}) J))
    exact IsColimit.ofIsoColimit this (Cocones.ext (Iso.refl c.pt) (fun {X} ↦ isEmptyElim X))
  /-
    J : Type v'
    inst✝³ : CategoryTheory.Category.{u', v'} J
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasStrictInitialObjects C
    inst✝ : IsEmpty J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    this : CategoryTheory.Limits.IsInitial c.pt
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  replace this := IsInitial.isVanKampenColimit this
  apply (IsVanKampenColimit.whiskerEquivalence_iff
    (equivalenceOfIsEmpty (Discrete PEmpty.{1}) J)).mp
  exact (this.precompose_isIso (Functor.uniqueFromEmpty
    ((equivalenceOfIsEmpty (Discrete PEmpty.{1}) J).functor ⋙ F)).hom).of_iso
    (Cocones.ext (Iso.refl _) (by simp))


theorem BinaryCofan.isVanKampen_iff (c : BinaryCofan X Y) :
    IsVanKampenColimit c ↔
      ∀ {X' Y' : C} (c' : BinaryCofan X' Y') (αX : X' ⟶ X) (αY : Y' ⟶ Y) (f : c'.pt ⟶ c.pt)
        (_ : αX ≫ c.inl = c'.inl ≫ f) (_ : αY ≫ c.inr = c'.inr ≫ f),
        Nonempty (IsColimit c') ↔ IsPullback c'.inl αX f c.inl ∧ IsPullback c'.inr αY f c.inr := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    c : CategoryTheory.Limits.BinaryCofan X Y
    ⊢ Iff (CategoryTheory.IsVanKampenColimit c) (∀ {X' Y' : C} (c' : CategoryTheor …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      ⊢ CategoryTheory.IsVanKampenColimit c → ∀ {X' Y' : C} (c' : CategoryTheory.Lim …
    -/
  · introv H hαX hαY
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : CategoryTheory.IsVanKampenColimit c
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hαX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Categor …
      hαY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Categor …
      ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (And (CategoryTheory.IsP …
    -/
    rw [H c' (mapPair αX αY) f (by ext ⟨⟨⟩⟩ <;> dsimp <;> assumption) (mapPair_equifibered _)]
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : CategoryTheory.IsVanKampenColimit c
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hαX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Categor …
      hαY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Categor …
      ⊢ Iff (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Cate …
    -/
    constructor
      /-
        case mp.mp
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        H : CategoryTheory.IsVanKampenColimit c
        X' Y' : C
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        f : Quiver.Hom c'.pt c.pt
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Categor …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Categor …
        ⊢ (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Category …
      -/
    · intro H
      /-
        case mp.mp
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        H✝ : CategoryTheory.IsVanKampenColimit c
        X' Y' : C
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        f : Quiver.Hom c'.pt c.pt
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Categor …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Categor …
        H : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Categor …
        ⊢ And (CategoryTheory.IsPullback c'.inl αX f c.inl) (CategoryTheory.IsPullback …
      -/
      exact ⟨H _, H _⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        H : CategoryTheory.IsVanKampenColimit c
        X' Y' : C
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        f : Quiver.Hom c'.pt c.pt
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Categor …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Categor …
        ⊢ And (CategoryTheory.IsPullback c'.inl αX f c.inl) (CategoryTheory.IsPullback …
      -/
    · rintro H ⟨⟨⟩⟩
      /-
        case mp.mpr.mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        H✝ : CategoryTheory.IsVanKampenColimit c
        X' Y' : C
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        f : Quiver.Hom c'.pt c.pt
        hαX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Categor …
        hαY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Categor …
        H : And (CategoryTheory.IsPullback c'.inl αX f c.inl) (CategoryTheory.IsPullba …
        ⊢ CategoryTheory.IsPullback (c'.ι.app { as := CategoryTheory.Limits.WalkingPai …
      -/
      exacts [H.1, H.2]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      ⊢ (∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver.H …
    -/
  · introv H F' hα h
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver. …
      F' : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Wal …
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (CategoryTheory.Limits.pair X Y)
      f : Quiver.Hom c'.pt c.pt
      hα : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStr …
      h : CategoryTheory.NatTrans.Equifibered α
      ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (∀ (j : CategoryTheory.D …
    -/
    let X' := F'.obj ⟨WalkingPair.left⟩
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver. …
      F' : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Wal …
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (CategoryTheory.Limits.pair X Y)
      f : Quiver.Hom c'.pt c.pt
      hα : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStr …
      h : CategoryTheory.NatTrans.Equifibered α
      X' : C := F'.obj { as := CategoryTheory.Limits.WalkingPair.left }
      ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (∀ (j : CategoryTheory.D …
    -/
    let Y' := F'.obj ⟨WalkingPair.right⟩
    have : F' = pair X' Y' := by
      apply Functor.hext
      · rintro ⟨⟨⟩⟩ <;> rfl
      · rintro ⟨⟨⟩⟩ ⟨j⟩ ⟨⟨rfl : _ = j⟩⟩ <;> simp [X', Y']
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver. …
      F' : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Wal …
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (CategoryTheory.Limits.pair X Y)
      f : Quiver.Hom c'.pt c.pt
      hα : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStr …
      h : CategoryTheory.NatTrans.Equifibered α
      X' : C := F'.obj { as := CategoryTheory.Limits.WalkingPair.left }
      Y' : C := F'.obj { as := CategoryTheory.Limits.WalkingPair.right }
      this : Eq F' (CategoryTheory.Limits.pair X' Y')
      ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (∀ (j : CategoryTheory.D …
    -/
    clear_value X' Y'
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver. …
      F' : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Wal …
      c' : CategoryTheory.Limits.Cocone F'
      α : Quiver.Hom F' (CategoryTheory.Limits.pair X Y)
      f : Quiver.Hom c'.pt c.pt
      hα : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStr …
      h : CategoryTheory.NatTrans.Equifibered α
      Y' X' : C
      this : Eq F' (CategoryTheory.Limits.pair X' Y')
      ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (∀ (j : CategoryTheory.D …
    -/
    subst this
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver. …
      Y' X' : C
      c' : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair X' Y')
      α : Quiver.Hom (CategoryTheory.Limits.pair X' Y') (CategoryTheory.Limits.pair  …
      f : Quiver.Hom c'.pt c.pt
      hα : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStr …
      h : CategoryTheory.NatTrans.Equifibered α
      ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (∀ (j : CategoryTheory.D …
    -/
    change BinaryCofan X' Y' at c'
    rw [H c' _ _ _ (NatTrans.congr_app hα ⟨WalkingPair.left⟩)
        (NatTrans.congr_app hα ⟨WalkingPair.right⟩)]
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver. …
      Y' X' : C
      α : Quiver.Hom (CategoryTheory.Limits.pair X' Y') (CategoryTheory.Limits.pair  …
      h : CategoryTheory.NatTrans.Equifibered α
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      f : Quiver.Hom c'.pt c.pt
      hα : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStr …
      ⊢ Iff (And (CategoryTheory.IsPullback c'.inl (α.app { as := CategoryTheory.Lim …
    -/
    constructor
      /-
        case mpr.mp
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        H : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver. …
        Y' X' : C
        α : Quiver.Hom (CategoryTheory.Limits.pair X' Y') (CategoryTheory.Limits.pair  …
        h : CategoryTheory.NatTrans.Equifibered α
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        f : Quiver.Hom c'.pt c.pt
        hα : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStr …
        ⊢ And (CategoryTheory.IsPullback c'.inl (α.app { as := CategoryTheory.Limits.W …
      -/
    · rintro H ⟨⟨⟩⟩
      /-
        case mpr.mp.mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        H✝ : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver …
        Y' X' : C
        α : Quiver.Hom (CategoryTheory.Limits.pair X' Y') (CategoryTheory.Limits.pair  …
        h : CategoryTheory.NatTrans.Equifibered α
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        f : Quiver.Hom c'.pt c.pt
        hα : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStr …
        H : And (CategoryTheory.IsPullback c'.inl (α.app { as := CategoryTheory.Limits …
        ⊢ CategoryTheory.IsPullback (c'.ι.app { as := CategoryTheory.Limits.WalkingPai …
      -/
      exacts [H.1, H.2]
      /-
        🎉 no goals
      -/
      /-
        case mpr.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        H : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver. …
        Y' X' : C
        α : Quiver.Hom (CategoryTheory.Limits.pair X' Y') (CategoryTheory.Limits.pair  …
        h : CategoryTheory.NatTrans.Equifibered α
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        f : Quiver.Hom c'.pt c.pt
        hα : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStr …
        ⊢ (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Category …
      -/
    · intro H
      /-
        case mpr.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        H✝ : ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver …
        Y' X' : C
        α : Quiver.Hom (CategoryTheory.Limits.pair X' Y') (CategoryTheory.Limits.pair  …
        h : CategoryTheory.NatTrans.Equifibered α
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        f : Quiver.Hom c'.pt c.pt
        hα : Eq (CategoryTheory.CategoryStruct.comp α c.ι) (CategoryTheory.CategoryStr …
        H : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Categor …
        ⊢ And (CategoryTheory.IsPullback c'.inl (α.app { as := CategoryTheory.Limits.W …
      -/
      exact ⟨H _, H _⟩
      /-
        🎉 no goals
      -/


theorem BinaryCofan.isVanKampen_mk {X Y : C} (c : BinaryCofan X Y)
    (cofans : ∀ X Y : C, BinaryCofan X Y) (colimits : ∀ X Y, IsColimit (cofans X Y))
    (cones : ∀ {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z), PullbackCone f g)
    (limits : ∀ {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z), IsLimit (cones f g))
    (h₁ : ∀ {X' Y' : C} (αX : X' ⟶ X) (αY : Y' ⟶ Y) (f : (cofans X' Y').pt ⟶ c.pt)
      (_ : αX ≫ c.inl = (cofans X' Y').inl ≫ f) (_ : αY ≫ c.inr = (cofans X' Y').inr ≫ f),
      IsPullback (cofans X' Y').inl αX f c.inl ∧ IsPullback (cofans X' Y').inr αY f c.inr)
    (h₂ : ∀ {Z : C} (f : Z ⟶ c.pt),
      IsColimit (BinaryCofan.mk (cones f c.inl).fst (cones f c.inr).fst)) :
    IsVanKampenColimit c := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    c : CategoryTheory.Limits.BinaryCofan X Y
    cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
    colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
    cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
    limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
    h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
    h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
    ⊢ CategoryTheory.IsVanKampenColimit c
  -/
  rw [BinaryCofan.isVanKampen_iff]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    c : CategoryTheory.Limits.BinaryCofan X Y
    cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
    colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
    cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
    limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
    h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
    h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
    ⊢ ∀ {X' Y' : C} (c' : CategoryTheory.Limits.BinaryCofan X' Y') (αX : Quiver.Ho …
  -/
  introv hX hY
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    c : CategoryTheory.Limits.BinaryCofan X Y
    cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
    colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
    cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
    limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
    h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
    h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
    X' Y' : C
    c' : CategoryTheory.Limits.BinaryCofan X' Y'
    αX : Quiver.Hom X' X
    αY : Quiver.Hom Y' Y
    f : Quiver.Hom c'.pt c.pt
    hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
    hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c')) (And (CategoryTheory.IsP …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
      colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
      cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
      limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
      h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
      h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
      hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
      ⊢ Nonempty (CategoryTheory.Limits.IsColimit c') → And (CategoryTheory.IsPullba …
    -/
  · rintro ⟨h⟩
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
      colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
      cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
      limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
      h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
      h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
      hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
      h : CategoryTheory.Limits.IsColimit c'
      ⊢ And (CategoryTheory.IsPullback c'.inl αX f c.inl) (CategoryTheory.IsPullback …
    -/
    let e := h.coconePointUniqueUpToIso (colimits _ _)
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
      colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
      cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
      limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
      h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
      h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
      hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
      h : CategoryTheory.Limits.IsColimit c'
      e : CategoryTheory.Iso c'.pt (cofans X' Y').pt := h.coconePointUniqueUpToIso ( …
      ⊢ And (CategoryTheory.IsPullback c'.inl αX f c.inl) (CategoryTheory.IsPullback …
    -/
    obtain ⟨hl, hr⟩ := h₁ αX αY (e.inv ≫ f) (by simp [e, hX]) (by simp [e, hY])
    /-
      case mp.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
      colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
      cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
      limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
      h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
      h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
      hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
      h : CategoryTheory.Limits.IsColimit c'
      e : CategoryTheory.Iso c'.pt (cofans X' Y').pt := h.coconePointUniqueUpToIso ( …
      hl : CategoryTheory.IsPullback (cofans X' Y').inl αX (CategoryTheory.CategoryS …
      hr : CategoryTheory.IsPullback (cofans X' Y').inr αY (CategoryTheory.CategoryS …
      ⊢ And (CategoryTheory.IsPullback c'.inl αX f c.inl) (CategoryTheory.IsPullback …
    -/
    constructor
      /-
        case mp.intro.intro.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
        colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
        cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
        limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
        h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
        h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
        X' Y' : C
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        f : Quiver.Hom c'.pt c.pt
        hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
        hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
        h : CategoryTheory.Limits.IsColimit c'
        e : CategoryTheory.Iso c'.pt (cofans X' Y').pt := h.coconePointUniqueUpToIso ( …
        hl : CategoryTheory.IsPullback (cofans X' Y').inl αX (CategoryTheory.CategoryS …
        hr : CategoryTheory.IsPullback (cofans X' Y').inr αY (CategoryTheory.CategoryS …
        ⊢ CategoryTheory.IsPullback c'.inl αX f c.inl
      -/
    · rw [← Category.id_comp αX, ← Iso.hom_inv_id_assoc e f]
      /-
        case mp.intro.intro.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
        colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
        cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
        limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
        h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
        h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
        X' Y' : C
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        f : Quiver.Hom c'.pt c.pt
        hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
        hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
        h : CategoryTheory.Limits.IsColimit c'
        e : CategoryTheory.Iso c'.pt (cofans X' Y').pt := h.coconePointUniqueUpToIso ( …
        hl : CategoryTheory.IsPullback (cofans X' Y').inl αX (CategoryTheory.CategoryS …
        hr : CategoryTheory.IsPullback (cofans X' Y').inr αY (CategoryTheory.CategoryS …
        ⊢ CategoryTheory.IsPullback c'.inl (CategoryTheory.CategoryStruct.comp (Catego …
      -/
      haveI : IsIso (𝟙 X') := inferInstance
      have : c'.inl ≫ e.hom = 𝟙 X' ≫ (cofans X' Y').inl := by
        dsimp [e]
        simp
      /-
        case mp.intro.intro.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
        colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
        cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
        limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
        h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
        h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
        X' Y' : C
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        f : Quiver.Hom c'.pt c.pt
        hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
        hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
        h : CategoryTheory.Limits.IsColimit c'
        e : CategoryTheory.Iso c'.pt (cofans X' Y').pt := h.coconePointUniqueUpToIso ( …
        hl : CategoryTheory.IsPullback (cofans X' Y').inl αX (CategoryTheory.CategoryS …
        hr : CategoryTheory.IsPullback (cofans X' Y').inr αY (CategoryTheory.CategoryS …
        this✝ : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.id X')
        this : Eq (CategoryTheory.CategoryStruct.comp c'.inl e.hom) (CategoryTheory.Ca …
        ⊢ CategoryTheory.IsPullback c'.inl (CategoryTheory.CategoryStruct.comp (Catego …
      -/
      exact (IsPullback.of_vert_isIso ⟨this⟩).paste_vert hl
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
        colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
        cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
        limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
        h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
        h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
        X' Y' : C
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        f : Quiver.Hom c'.pt c.pt
        hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
        hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
        h : CategoryTheory.Limits.IsColimit c'
        e : CategoryTheory.Iso c'.pt (cofans X' Y').pt := h.coconePointUniqueUpToIso ( …
        hl : CategoryTheory.IsPullback (cofans X' Y').inl αX (CategoryTheory.CategoryS …
        hr : CategoryTheory.IsPullback (cofans X' Y').inr αY (CategoryTheory.CategoryS …
        ⊢ CategoryTheory.IsPullback c'.inr αY f c.inr
      -/
    · rw [← Category.id_comp αY, ← Iso.hom_inv_id_assoc e f]
      /-
        case mp.intro.intro.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
        colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
        cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
        limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
        h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
        h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
        X' Y' : C
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        f : Quiver.Hom c'.pt c.pt
        hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
        hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
        h : CategoryTheory.Limits.IsColimit c'
        e : CategoryTheory.Iso c'.pt (cofans X' Y').pt := h.coconePointUniqueUpToIso ( …
        hl : CategoryTheory.IsPullback (cofans X' Y').inl αX (CategoryTheory.CategoryS …
        hr : CategoryTheory.IsPullback (cofans X' Y').inr αY (CategoryTheory.CategoryS …
        ⊢ CategoryTheory.IsPullback c'.inr (CategoryTheory.CategoryStruct.comp (Catego …
      -/
      haveI : IsIso (𝟙 Y') := inferInstance
      have : c'.inr ≫ e.hom = 𝟙 Y' ≫ (cofans X' Y').inr := by
        dsimp [e]
        simp
      /-
        case mp.intro.intro.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
        colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
        cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
        limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
        h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
        h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
        X' Y' : C
        c' : CategoryTheory.Limits.BinaryCofan X' Y'
        αX : Quiver.Hom X' X
        αY : Quiver.Hom Y' Y
        f : Quiver.Hom c'.pt c.pt
        hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
        hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
        h : CategoryTheory.Limits.IsColimit c'
        e : CategoryTheory.Iso c'.pt (cofans X' Y').pt := h.coconePointUniqueUpToIso ( …
        hl : CategoryTheory.IsPullback (cofans X' Y').inl αX (CategoryTheory.CategoryS …
        hr : CategoryTheory.IsPullback (cofans X' Y').inr αY (CategoryTheory.CategoryS …
        this✝ : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.id Y')
        this : Eq (CategoryTheory.CategoryStruct.comp c'.inr e.hom) (CategoryTheory.Ca …
        ⊢ CategoryTheory.IsPullback c'.inr (CategoryTheory.CategoryStruct.comp (Catego …
      -/
      exact (IsPullback.of_vert_isIso ⟨this⟩).paste_vert hr
      /-
        🎉 no goals
      -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
      colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
      cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
      limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
      h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
      h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
      hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
      ⊢ And (CategoryTheory.IsPullback c'.inl αX f c.inl) (CategoryTheory.IsPullback …
    -/
  · rintro ⟨H₁, H₂⟩
    /-
      case mpr.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
      colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
      cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
      limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
      h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
      h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
      hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
      H₁ : CategoryTheory.IsPullback c'.inl αX f c.inl
      H₂ : CategoryTheory.IsPullback c'.inr αY f c.inr
      ⊢ Nonempty (CategoryTheory.Limits.IsColimit c')
    -/
    refine ⟨IsColimit.ofIsoColimit ?_ <| (isoBinaryCofanMk _).symm⟩
    /-
      case mpr.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
      colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
      cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
      limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
      h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
      h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
      hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
      H₁ : CategoryTheory.IsPullback c'.inl αX f c.inl
      H₂ : CategoryTheory.IsPullback c'.inr αY f c.inr
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk c'.inl …
    -/
    let e₁ : X' ≅ _ := H₁.isLimit.conePointUniqueUpToIso (limits _ _)
    /-
      case mpr.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
      colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
      cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
      limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
      h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
      h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
      hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
      H₁ : CategoryTheory.IsPullback c'.inl αX f c.inl
      H₂ : CategoryTheory.IsPullback c'.inr αY f c.inr
      e₁ : CategoryTheory.Iso X' (cones f c.inl).pt := H₁.isLimit.conePointUniqueUpT …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk c'.inl …
    -/
    let e₂ : Y' ≅ _ := H₂.isLimit.conePointUniqueUpToIso (limits _ _)
    /-
      case mpr.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
      colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
      cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
      limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
      h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
      h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
      hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
      H₁ : CategoryTheory.IsPullback c'.inl αX f c.inl
      H₂ : CategoryTheory.IsPullback c'.inr αY f c.inr
      e₁ : CategoryTheory.Iso X' (cones f c.inl).pt := H₁.isLimit.conePointUniqueUpT …
      e₂ : CategoryTheory.Iso Y' (cones f c.inr).pt := H₂.isLimit.conePointUniqueUpT …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk c'.inl …
    -/
    have he₁ : c'.inl = e₁.hom ≫ (cones f c.inl).fst := by simp [e₁]
    /-
      case mpr.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
      colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
      cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
      limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
      h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
      h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
      hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
      H₁ : CategoryTheory.IsPullback c'.inl αX f c.inl
      H₂ : CategoryTheory.IsPullback c'.inr αY f c.inr
      e₁ : CategoryTheory.Iso X' (cones f c.inl).pt := H₁.isLimit.conePointUniqueUpT …
      e₂ : CategoryTheory.Iso Y' (cones f c.inr).pt := H₂.isLimit.conePointUniqueUpT …
      he₁ : Eq c'.inl (CategoryTheory.CategoryStruct.comp e₁.hom (cones f c.inl).fst)
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk c'.inl …
    -/
    have he₂ : c'.inr = e₂.hom ≫ (cones f c.inr).fst := by simp [e₂]
    /-
      case mpr.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      cofans : (X Y : C) → CategoryTheory.Limits.BinaryCofan X Y
      colimits : (X Y : C) → CategoryTheory.Limits.IsColimit (cofans X Y)
      cones : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryTh …
      limits : {X Y Z : C} → (f : Quiver.Hom X Z) → (g : Quiver.Hom Y Z) → CategoryT …
      h₁ : ∀ {X' Y' : C} (αX : Quiver.Hom X' X) (αY : Quiver.Hom Y' Y) (f : Quiver.H …
      h₂ : {Z : C} → (f : Quiver.Hom Z c.pt) → CategoryTheory.Limits.IsColimit (Cate …
      X' Y' : C
      c' : CategoryTheory.Limits.BinaryCofan X' Y'
      αX : Quiver.Hom X' X
      αY : Quiver.Hom Y' Y
      f : Quiver.Hom c'.pt c.pt
      hX : Eq (CategoryTheory.CategoryStruct.comp αX c.inl) (CategoryTheory.Category …
      hY : Eq (CategoryTheory.CategoryStruct.comp αY c.inr) (CategoryTheory.Category …
      H₁ : CategoryTheory.IsPullback c'.inl αX f c.inl
      H₂ : CategoryTheory.IsPullback c'.inr αY f c.inr
      e₁ : CategoryTheory.Iso X' (cones f c.inl).pt := H₁.isLimit.conePointUniqueUpT …
      e₂ : CategoryTheory.Iso Y' (cones f c.inr).pt := H₂.isLimit.conePointUniqueUpT …
      he₁ : Eq c'.inl (CategoryTheory.CategoryStruct.comp e₁.hom (cones f c.inl).fst)
      he₂ : Eq c'.inr (CategoryTheory.CategoryStruct.comp e₂.hom (cones f c.inr).fst)
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk c'.inl …
    -/
    rw [he₁, he₂]
    exact (BinaryCofan.mk _ _).isColimitCompRightIso e₂.hom
      ((BinaryCofan.mk _ _).isColimitCompLeftIso e₁.hom (h₂ f))


theorem BinaryCofan.mono_inr_of_isVanKampen [HasInitial C] {X Y : C} {c : BinaryCofan X Y}
    (h : IsVanKampenColimit c) : Mono c.inr := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasInitial C
    X Y : C
    c : CategoryTheory.Limits.BinaryCofan X Y
    h : CategoryTheory.IsVanKampenColimit c
    ⊢ CategoryTheory.Mono c.inr
  -/
  refine PullbackCone.mono_of_isLimitMkIdId _ (IsPullback.isLimit ?_)
  refine (h (BinaryCofan.mk (initial.to Y) (𝟙 Y)) (mapPair (initial.to X) (𝟙 Y)) c.inr ?_
      (mapPair_equifibered _)).mp ⟨?_⟩ ⟨WalkingPair.right⟩
    /-
      case refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasInitial C
      X Y : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      h : CategoryTheory.IsVanKampenColimit c
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.mapPair (Categ …
    -/
                 /-
                   🎉 no goals
                 -/
  · ext ⟨⟨⟩⟩ <;> dsimp; simp
                        /-
                          🎉 no goals
                        -/
  · exact ((BinaryCofan.isColimit_iff_isIso_inr initialIsInitial _).mpr (by
      dsimp
      infer_instance)).some


theorem BinaryCofan.isPullback_initial_to_of_isVanKampen [HasInitial C] {c : BinaryCofan X Y}
    (h : IsVanKampenColimit c) : IsPullback (initial.to _) (initial.to _) c.inl c.inr := by
  refine ((h (BinaryCofan.mk (initial.to Y) (𝟙 Y)) (mapPair (initial.to X) (𝟙 Y)) c.inr ?_
      (mapPair_equifibered _)).mp ⟨?_⟩ ⟨WalkingPair.left⟩).flip
    /-
      case refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasInitial C
      c : CategoryTheory.Limits.BinaryCofan X Y
      h : CategoryTheory.IsVanKampenColimit c
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.mapPair (Categ …
    -/
                 /-
                   🎉 no goals
                 -/
  · ext ⟨⟨⟩⟩ <;> dsimp; simp
                        /-
                          🎉 no goals
                        -/
  · exact ((BinaryCofan.isColimit_iff_isIso_inr initialIsInitial _).mpr (by
      dsimp
      infer_instance)).some


theorem isUniversalColimit_extendCofan {n : ℕ} (f : Fin (n + 1) → C)
    {c₁ : Cofan fun i : Fin n ↦ f i.succ} {c₂ : BinaryCofan (f 0) c₁.pt}
    (t₁ : IsUniversalColimit c₁) (t₂ : IsUniversalColimit c₂)
    [∀ {Z} (i : Z ⟶ c₂.pt), HasPullback c₂.inr i] :
    IsUniversalColimit (extendCofan c₁ c₂) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsUniversalColimit c₁
    t₂ : CategoryTheory.IsUniversalColimit c₂
    inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
    ⊢ CategoryTheory.IsUniversalColimit (CategoryTheory.extendCofan c₁ c₂)
  -/
  intro F c α i e hα H
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsUniversalColimit c₁
    t₂ : CategoryTheory.IsUniversalColimit c₂
    inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
    c : CategoryTheory.Limits.Cocone F
    α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c)
  -/
  let F' : Fin (n + 1) → C := F.obj ∘ Discrete.mk
  have : F = Discrete.functor F' := by
    apply Functor.hext
    · exact fun i ↦ rfl
    · rintro ⟨i⟩ ⟨j⟩ ⟨⟨rfl : i = j⟩⟩
      simp [F']
  have t₁' := @t₁ (Discrete.functor (fun j ↦ F.obj ⟨j.succ⟩))
    (Cofan.mk (pullback c₂.inr i) fun j ↦ pullback.lift (α.app _ ≫ c₁.inj _) (c.ι.app _) ?_)
    (Discrete.natTrans fun i ↦ α.app _) (pullback.fst _ _) ?_
    (NatTrans.equifibered_of_discrete _) ?_
  /-
    case refine_4
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsUniversalColimit c₁
    t₂ : CategoryTheory.IsUniversalColimit c₂
    inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
    c : CategoryTheory.Limits.Cocone F
    α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
    F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
    this : Eq F (CategoryTheory.Discrete.functor F')
    t₁' : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.m …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c)
  -/
  rotate_left
  · simpa only [Functor.const_obj_obj, pair_obj_right, Discrete.functor_obj, Category.assoc,
      extendCofan_pt, Functor.const_obj_obj, NatTrans.comp_app, extendCofan_ι_app,
      Fin.cases_succ, Functor.const_map_app] using congr_app e ⟨j.succ⟩
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsUniversalColimit c₁
      t₂ : CategoryTheory.IsUniversalColimit c₂
      inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
      F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
      this : Eq F (CategoryTheory.Discrete.functor F')
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Discrete.natTrans fun …
    -/
  · ext j
    /-
      case refine_2.w.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsUniversalColimit c₁
      t₂ : CategoryTheory.IsUniversalColimit c₂
      inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
      F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
      this : Eq F (CategoryTheory.Discrete.functor F')
      j : CategoryTheory.Discrete (Fin n)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Discrete.natTrans fu …
    -/
    dsimp
    /-
      case refine_2.w.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsUniversalColimit c₁
      t₂ : CategoryTheory.IsUniversalColimit c₂
      inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
      F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
      this : Eq F (CategoryTheory.Discrete.functor F')
      j : CategoryTheory.Discrete (Fin n)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app { as := j.as.succ }) (c₁.ι.app …
    -/
    simp only [limit.lift_π, PullbackCone.mk_pt, PullbackCone.mk_π_app, Cofan.inj]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsUniversalColimit c₁
      t₂ : CategoryTheory.IsUniversalColimit c₂
      inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
      F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
      this : Eq F (CategoryTheory.Discrete.functor F')
      ⊢ ∀ (j : CategoryTheory.Discrete (Fin n)), CategoryTheory.IsPullback ((Categor …
    -/
  · intro j
    simp only [pair_obj_right, Functor.const_obj_obj, Discrete.functor_obj, id_eq,
      extendCofan_pt, eq_mpr_eq_cast, Cofan.mk_pt, Cofan.mk_ι_app, Discrete.natTrans_app]
    /-
      case refine_3
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsUniversalColimit c₁
      t₂ : CategoryTheory.IsUniversalColimit c₂
      inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
      F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
      this : Eq F (CategoryTheory.Discrete.functor F')
      j : CategoryTheory.Discrete (Fin n)
      ⊢ CategoryTheory.IsPullback (CategoryTheory.Limits.pullback.lift (CategoryTheo …
    -/
    refine IsPullback.of_right ?_ ?_ (IsPullback.of_hasPullback (BinaryCofan.inr c₂) i).flip
    · simp only [Functor.const_obj_obj, pair_obj_right, limit.lift_π,
        PullbackCone.mk_pt, PullbackCone.mk_π_app]
      /-
        case refine_3.refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsUniversalColimit c₁
        t₂ : CategoryTheory.IsUniversalColimit c₂
        inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
        F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
        c : CategoryTheory.Limits.Cocone F
        α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
        F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
        this : Eq F (CategoryTheory.Discrete.functor F')
        j : CategoryTheory.Discrete (Fin n)
        ⊢ CategoryTheory.IsPullback (c.ι.app { as := j.as.succ }) (α.app { as := j.as. …
      -/
      exact H _
      /-
        🎉 no goals
      -/
      /-
        case refine_3.refine_2
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsUniversalColimit c₁
        t₂ : CategoryTheory.IsUniversalColimit c₂
        inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
        F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
        c : CategoryTheory.Limits.Cocone F
        α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
        F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
        this : Eq F (CategoryTheory.Discrete.functor F')
        j : CategoryTheory.Discrete (Fin n)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
      -/
    · simp only [limit.lift_π, PullbackCone.mk_pt, PullbackCone.mk_π_app, Cofan.inj]
      /-
        🎉 no goals
      -/
  /-
    case refine_4
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsUniversalColimit c₁
    t₂ : CategoryTheory.IsUniversalColimit c₂
    inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
    c : CategoryTheory.Limits.Cocone F
    α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
    F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
    this : Eq F (CategoryTheory.Discrete.functor F')
    t₁' : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.m …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c)
  -/
  obtain ⟨H₁⟩ := t₁'
  have t₂' := @t₂ (pair (F.obj ⟨0⟩) (pullback c₂.inr i))
    (BinaryCofan.mk (c.ι.app ⟨0⟩) (pullback.snd _ _)) (mapPair (α.app _) (pullback.fst _ _)) i ?_
    (mapPair_equifibered _) ?_
  /-
    case refine_4.intro.refine_3
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsUniversalColimit c₁
    t₂ : CategoryTheory.IsUniversalColimit c₂
    inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
    c : CategoryTheory.Limits.Cocone F
    α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
    F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
    this : Eq F (CategoryTheory.Discrete.functor F')
    H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
    t₂' : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryC …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c)
  -/
  rotate_left
    /-
      case refine_4.intro.refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsUniversalColimit c₁
      t₂ : CategoryTheory.IsUniversalColimit c₂
      inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
      F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
      this : Eq F (CategoryTheory.Discrete.functor F')
      H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.mapPair (α.app …
    -/
  · ext ⟨⟨⟩⟩
      /-
        case refine_4.intro.refine_1.w.h.mk.left
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsUniversalColimit c₁
        t₂ : CategoryTheory.IsUniversalColimit c₂
        inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
        F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
        c : CategoryTheory.Limits.Cocone F
        α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
        F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
        this : Eq F (CategoryTheory.Discrete.functor F')
        H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.mapPair (α.ap …
      -/
    · simpa [mapPair] using congr_app e ⟨0⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_4.intro.refine_1.w.h.mk.right
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsUniversalColimit c₁
        t₂ : CategoryTheory.IsUniversalColimit c₂
        inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
        F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
        c : CategoryTheory.Limits.Cocone F
        α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
        F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
        this : Eq F (CategoryTheory.Discrete.functor F')
        H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.mapPair (α.ap …
      -/
    · simpa using pullback.condition
      /-
        🎉 no goals
      -/
    /-
      case refine_4.intro.refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsUniversalColimit c₁
      t₂ : CategoryTheory.IsUniversalColimit c₂
      inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
      F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
      this : Eq F (CategoryTheory.Discrete.functor F')
      H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), CategoryT …
    -/
  · rintro ⟨⟨⟩⟩
    · simp only [pair_obj_right, Functor.const_obj_obj, pair_obj_left, BinaryCofan.mk_pt,
        BinaryCofan.ι_app_left, BinaryCofan.mk_inl, mapPair_left]
      /-
        case refine_4.intro.refine_2.mk.left
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsUniversalColimit c₁
        t₂ : CategoryTheory.IsUniversalColimit c₂
        inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
        F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
        c : CategoryTheory.Limits.Cocone F
        α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
        F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
        this : Eq F (CategoryTheory.Discrete.functor F')
        H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
        ⊢ CategoryTheory.IsPullback (c.ι.app { as := 0 }) (α.app { as := 0 }) i c₂.inl
      -/
      exact H ⟨0⟩
      /-
        🎉 no goals
      -/
    · simp only [pair_obj_right, Functor.const_obj_obj, BinaryCofan.mk_pt, BinaryCofan.ι_app_right,
        BinaryCofan.mk_inr, mapPair_right]
      /-
        case refine_4.intro.refine_2.mk.right
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsUniversalColimit c₁
        t₂ : CategoryTheory.IsUniversalColimit c₂
        inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
        F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
        c : CategoryTheory.Limits.Cocone F
        α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
        F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
        this : Eq F (CategoryTheory.Discrete.functor F')
        H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
        ⊢ CategoryTheory.IsPullback (CategoryTheory.Limits.pullback.snd c₂.inr i) (Cat …
      -/
      exact (IsPullback.of_hasPullback (BinaryCofan.inr c₂) i).flip
      /-
        🎉 no goals
      -/
  /-
    case refine_4.intro.refine_3
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsUniversalColimit c₁
    t₂ : CategoryTheory.IsUniversalColimit c₂
    inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
    c : CategoryTheory.Limits.Cocone F
    α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
    F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
    this : Eq F (CategoryTheory.Discrete.functor F')
    H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
    t₂' : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryC …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c)
  -/
  obtain ⟨H₂⟩ := t₂'
  /-
    case refine_4.intro.refine_3.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsUniversalColimit c₁
    t₂ : CategoryTheory.IsUniversalColimit c₂
    inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
    c : CategoryTheory.Limits.Cocone F
    α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
    F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
    this : Eq F (CategoryTheory.Discrete.functor F')
    H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (c. …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c)
  -/
  clear_value F'
  /-
    case refine_4.intro.refine_3.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsUniversalColimit c₁
    t₂ : CategoryTheory.IsUniversalColimit c₂
    inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
    c : CategoryTheory.Limits.Cocone F
    α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
    H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (c. …
    F' : Fin (HAdd.hAdd n 1) → C
    this : Eq F (CategoryTheory.Discrete.functor F')
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c)
  -/
  subst this
  refine ⟨IsColimit.ofIsoColimit (extendCofanIsColimit
    (fun i ↦ (Discrete.functor F').obj ⟨i⟩) H₁ H₂) <| Cocones.ext (Iso.refl _) ?_⟩
  /-
    case refine_4.intro.refine_3.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsUniversalColimit c₁
    t₂ : CategoryTheory.IsUniversalColimit c₂
    inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
    F' : Fin (HAdd.hAdd n 1) → C
    c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
    α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
    H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (c. …
    ⊢ ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory.Ca …
  -/
  dsimp
  /-
    case refine_4.intro.refine_3.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsUniversalColimit c₁
    t₂ : CategoryTheory.IsUniversalColimit c₂
    inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
    F' : Fin (HAdd.hAdd n 1) → C
    c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
    α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
    H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (c. …
    ⊢ ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory.Ca …
  -/
  rintro ⟨j⟩
  simp only [Discrete.functor_obj, limit.lift_π, PullbackCone.mk_pt,
    PullbackCone.mk_π_app, Category.comp_id]
  /-
    case refine_4.intro.refine_3.intro.mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsUniversalColimit c₁
    t₂ : CategoryTheory.IsUniversalColimit c₂
    inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
    F' : Fin (HAdd.hAdd n 1) → C
    c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
    α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
    H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
    H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (c. …
    j : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Fin.cases (c.ι.app { as := 0 }) (fun i => c.ι.app { as := i.succ }) j) ( …
  -/
  induction' j using Fin.inductionOn
    /-
      case refine_4.intro.refine_3.intro.mk.zero
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsUniversalColimit c₁
      t₂ : CategoryTheory.IsUniversalColimit c₂
      inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
      F' : Fin (HAdd.hAdd n 1) → C
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
      α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
      H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
      H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (c. …
      ⊢ Eq (Fin.cases (c.ι.app { as := 0 }) (fun i => c.ι.app { as := i.succ }) 0) ( …
    -/
  · simp only [Fin.cases_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_4.intro.refine_3.intro.mk.succ
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsUniversalColimit c₁
      t₂ : CategoryTheory.IsUniversalColimit c₂
      inst✝ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback  …
      F' : Fin (HAdd.hAdd n 1) → C
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
      α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      H : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), CategoryTheory.IsPu …
      H₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (Category …
      H₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (c. …
      i✝ : Fin n
      a✝ : Eq (Fin.cases (c.ι.app { as := 0 }) (fun i => c.ι.app { as := i.succ }) i …
      ⊢ Eq (Fin.cases (c.ι.app { as := 0 }) (fun i => c.ι.app { as := i.succ }) i✝.s …
    -/
  · simp only [Fin.cases_succ]
    /-
      🎉 no goals
    -/


theorem isVanKampenColimit_extendCofan {n : ℕ} (f : Fin (n + 1) → C)
    {c₁ : Cofan fun i : Fin n ↦ f i.succ} {c₂ : BinaryCofan (f 0) c₁.pt}
    (t₁ : IsVanKampenColimit c₁) (t₂ : IsVanKampenColimit c₂)
    [∀ {Z} (i : Z ⟶ c₂.pt), HasPullback c₂.inr i]
    [HasFiniteCoproducts C] :
    IsVanKampenColimit (extendCofan c₁ c₂) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsVanKampenColimit c₁
    t₂ : CategoryTheory.IsVanKampenColimit c₂
    inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    ⊢ CategoryTheory.IsVanKampenColimit (CategoryTheory.extendCofan c₁ c₂)
  -/
  intro F c α i e hα
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsVanKampenColimit c₁
    t₂ : CategoryTheory.IsVanKampenColimit c₂
    inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
    c : CategoryTheory.Limits.Cocone F
    α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c)) (∀ (j : CategoryTheory.Di …
  -/
  refine ⟨?_, isUniversalColimit_extendCofan f t₁.isUniversal t₂.isUniversal c α i e hα⟩
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsVanKampenColimit c₁
    t₂ : CategoryTheory.IsVanKampenColimit c₂
    inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
    c : CategoryTheory.Limits.Cocone F
    α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit c) → ∀ (j : CategoryTheory.Discret …
  -/
  intro ⟨Hc⟩ ⟨j⟩
  have t₂' := (@t₂ (pair (F.obj ⟨0⟩) (∐ fun (j : Fin n) ↦ F.obj ⟨j.succ⟩))
    (BinaryCofan.mk (P := c.pt) (c.ι.app _) (Sigma.desc fun b ↦ c.ι.app _))
    (mapPair (α.app _) (Sigma.desc fun b ↦ α.app _ ≫ c₁.inj _)) i ?_
    (mapPair_equifibered _)).mp ⟨?_⟩
  /-
    case refine_3
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsVanKampenColimit c₁
    t₂ : CategoryTheory.IsVanKampenColimit c₂
    inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
    c : CategoryTheory.Limits.Cocone F
    α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    Hc : CategoryTheory.Limits.IsColimit c
    j : Fin (HAdd.hAdd n 1)
    t₂' : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Categ …
    ⊢ CategoryTheory.IsPullback (c.ι.app { as := j }) (α.app { as := j }) i ((Cate …
  -/
  rotate_left
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsVanKampenColimit c₁
      t₂ : CategoryTheory.IsVanKampenColimit c₂
      inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      Hc : CategoryTheory.Limits.IsColimit c
      j : Fin (HAdd.hAdd n 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.mapPair (α.app …
    -/
  · ext ⟨⟨⟩⟩
    · simpa only [pair_obj_left, Functor.const_obj_obj, pair_obj_right, Discrete.functor_obj,
        NatTrans.comp_app, mapPair_left, BinaryCofan.ι_app_left, BinaryCofan.mk_pt,
        BinaryCofan.mk_inl, Functor.const_map_app, extendCofan_pt,
        extendCofan_ι_app, Fin.cases_zero] using congr_app e ⟨0⟩
      /-
        case refine_1.w.h.mk.right
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
        c : CategoryTheory.Limits.Cocone F
        α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        j : Fin (HAdd.hAdd n 1)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.mapPair (α.ap …
      -/
    · dsimp
      /-
        case refine_1.w.h.mk.right
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
        c : CategoryTheory.Limits.Cocone F
        α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        j : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc fun …
      -/
      ext j
      simpa only [colimit.ι_desc_assoc, Discrete.functor_obj, Cofan.mk_pt, Cofan.mk_ι_app,
        Category.assoc, extendCofan_pt, Functor.const_obj_obj, NatTrans.comp_app, extendCofan_ι_app,
        Fin.cases_succ, Functor.const_map_app] using congr_app e ⟨j.succ⟩
    /-
      case refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsVanKampenColimit c₁
      t₂ : CategoryTheory.IsVanKampenColimit c₂
      inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      Hc : CategoryTheory.Limits.IsColimit c
      j : Fin (HAdd.hAdd n 1)
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (c.ι.a …
    -/
  · let F' : Fin (n + 1) → C := F.obj ∘ Discrete.mk
    have : F = Discrete.functor F' := by
      apply Functor.hext
      · exact fun i ↦ rfl
      · rintro ⟨i⟩ ⟨j⟩ ⟨⟨rfl : i = j⟩⟩
        simp [F']
    /-
      case refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsVanKampenColimit c₁
      t₂ : CategoryTheory.IsVanKampenColimit c₂
      inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      Hc : CategoryTheory.Limits.IsColimit c
      j : Fin (HAdd.hAdd n 1)
      F' : Fin (HAdd.hAdd n 1) → C := Function.comp F.obj CategoryTheory.Discrete.mk
      this : Eq F (CategoryTheory.Discrete.functor F')
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (c.ι.a …
    -/
    clear_value F'
    /-
      case refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsVanKampenColimit c₁
      t₂ : CategoryTheory.IsVanKampenColimit c₂
      inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      Hc : CategoryTheory.Limits.IsColimit c
      j : Fin (HAdd.hAdd n 1)
      F' : Fin (HAdd.hAdd n 1) → C
      this : Eq F (CategoryTheory.Discrete.functor F')
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (c.ι.a …
    -/
    subst this
    apply BinaryCofan.IsColimit.mk _ (fun {T} f₁ f₂ ↦ Hc.desc (Cofan.mk T (Fin.cases f₁
      (fun i ↦ Sigma.ι (fun (j : Fin n) ↦ (Discrete.functor F').obj ⟨j.succ⟩) _ ≫ f₂))))
      /-
        case refine_2.hd₁
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        j : Fin (HAdd.hAdd n 1)
        F' : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
        α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        ⊢ ∀ {T : C} (f : Quiver.Hom (F' 0) T) (g : Quiver.Hom (CategoryTheory.Limits.s …
      -/
    · intro T f₁ f₂
      simp only [Discrete.functor_obj, pair_obj_left, BinaryCofan.mk_pt, Functor.const_obj_obj,
        BinaryCofan.ι_app_left, BinaryCofan.mk_inl, IsColimit.fac, Cofan.mk_pt, Cofan.mk_ι_app,
        Fin.cases_zero]
      /-
        case refine_2.hd₂
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        j : Fin (HAdd.hAdd n 1)
        F' : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
        α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        ⊢ ∀ {T : C} (f : Quiver.Hom (F' 0) T) (g : Quiver.Hom (CategoryTheory.Limits.s …
      -/
    · intro T f₁ f₂
      simp only [Discrete.functor_obj, pair_obj_right, BinaryCofan.mk_pt, Functor.const_obj_obj,
        BinaryCofan.ι_app_right, BinaryCofan.mk_inr]
      /-
        case refine_2.hd₂
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        j : Fin (HAdd.hAdd n 1)
        F' : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
        α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        T : C
        f₁ : Quiver.Hom (F' 0) T
        f₂ : Quiver.Hom (CategoryTheory.Limits.sigmaObj fun j => (CategoryTheory.Discr …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc fun …
      -/
      ext j
      simp only [colimit.ι_desc_assoc, Discrete.functor_obj, Cofan.mk_pt,
        Cofan.mk_ι_app, IsColimit.fac, Fin.cases_succ]
      /-
        case refine_2.uniq
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        j : Fin (HAdd.hAdd n 1)
        F' : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
        α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        ⊢ ∀ {T : C} (f : Quiver.Hom (F' 0) T) (g : Quiver.Hom (CategoryTheory.Limits.s …
      -/
    · intro T f₁ f₂ f₃ m₁ m₂
      /-
        case refine_2.uniq
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        j : Fin (HAdd.hAdd n 1)
        F' : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
        α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        T : C
        f₁ : Quiver.Hom (F' 0) T
        f₂ : Quiver.Hom (CategoryTheory.Limits.sigmaObj fun j => (CategoryTheory.Discr …
        f₃ : Quiver.Hom (CategoryTheory.Limits.BinaryCofan.mk (c.ι.app { as := 0 }) (C …
        m₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        m₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        ⊢ Eq f₃ (Hc.desc (CategoryTheory.Limits.Cofan.mk T fun i => Fin.cases f₁ (fun  …
      -/
      simp at m₁ m₂ ⊢
      refine Hc.uniq (Cofan.mk T (Fin.cases f₁
        (fun i ↦ Sigma.ι (fun (j : Fin n) ↦ (Discrete.functor F').obj ⟨j.succ⟩) _ ≫ f₂))) _ ?_
      /-
        case refine_2.uniq
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        j : Fin (HAdd.hAdd n 1)
        F' : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
        α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        T : C
        f₁ : Quiver.Hom (F' 0) T
        f₂ : Quiver.Hom (CategoryTheory.Limits.sigmaObj fun j => (CategoryTheory.Discr …
        f₃ : Quiver.Hom (CategoryTheory.Limits.BinaryCofan.mk (c.ι.app { as := 0 }) (C …
        m₁ : Eq (CategoryTheory.CategoryStruct.comp (c.ι.app { as := 0 }) f₃) f₁
        m₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc  …
        ⊢ ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory.Ca …
      -/
      intro ⟨j⟩
      /-
        case refine_2.uniq
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        j✝ : Fin (HAdd.hAdd n 1)
        F' : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
        α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        T : C
        f₁ : Quiver.Hom (F' 0) T
        f₂ : Quiver.Hom (CategoryTheory.Limits.sigmaObj fun j => (CategoryTheory.Discr …
        f₃ : Quiver.Hom (CategoryTheory.Limits.BinaryCofan.mk (c.ι.app { as := 0 }) (C …
        m₁ : Eq (CategoryTheory.CategoryStruct.comp (c.ι.app { as := 0 }) f₃) f₁
        m₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc  …
        j : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app { as := j }) f₃) ((CategoryT …
      -/
      simp only [Discrete.functor_obj, Cofan.mk_pt, Functor.const_obj_obj, Cofan.mk_ι_app]
      /-
        case refine_2.uniq
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        j✝ : Fin (HAdd.hAdd n 1)
        F' : Fin (HAdd.hAdd n 1) → C
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
        α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        T : C
        f₁ : Quiver.Hom (F' 0) T
        f₂ : Quiver.Hom (CategoryTheory.Limits.sigmaObj fun j => (CategoryTheory.Discr …
        f₃ : Quiver.Hom (CategoryTheory.Limits.BinaryCofan.mk (c.ι.app { as := 0 }) (C …
        m₁ : Eq (CategoryTheory.CategoryStruct.comp (c.ι.app { as := 0 }) f₃) f₁
        m₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc  …
        j : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app { as := j }) f₃) (Fin.cases  …
      -/
      induction' j using Fin.inductionOn with j _
        /-
          case refine_2.uniq.zero
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          n : Nat
          f : Fin (HAdd.hAdd n 1) → C
          c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
          c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
          t₁ : CategoryTheory.IsVanKampenColimit c₁
          t₂ : CategoryTheory.IsVanKampenColimit c₂
          inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          j : Fin (HAdd.hAdd n 1)
          F' : Fin (HAdd.hAdd n 1) → C
          c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F')
          α : Quiver.Hom (CategoryTheory.Discrete.functor F') (CategoryTheory.Discrete.f …
          i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
          e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
          hα : CategoryTheory.NatTrans.Equifibered α
          Hc : CategoryTheory.Limits.IsColimit c
          T : C
          f₁ : Quiver.Hom (F' 0) T
          f₂ : Quiver.Hom (CategoryTheory.Limits.sigmaObj fun j => (CategoryTheory.Discr …
          f₃ : Quiver.Hom (CategoryTheory.Limits.BinaryCofan.mk (c.ι.app { as := 0 }) (C …
          m₁ : Eq (CategoryTheory.CategoryStruct.comp (c.ι.app { as := 0 }) f₃) f₁
          m₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app { as := 0 }) f₃) (Fin.cases  …
        -/
      · simp only [Fin.cases_zero, m₁]
        /-
          🎉 no goals
        -/
      · simp only [← m₂, colimit.ι_desc_assoc, Discrete.functor_obj,
          Cofan.mk_pt, Cofan.mk_ι_app, Fin.cases_succ]
  /-
    case refine_3
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    n : Nat
    f : Fin (HAdd.hAdd n 1) → C
    c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
    c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
    t₁ : CategoryTheory.IsVanKampenColimit c₁
    t₂ : CategoryTheory.IsVanKampenColimit c₂
    inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
    c : CategoryTheory.Limits.Cocone F
    α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
    i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
    e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
    hα : CategoryTheory.NatTrans.Equifibered α
    Hc : CategoryTheory.Limits.IsColimit c
    j : Fin (HAdd.hAdd n 1)
    t₂' : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Categ …
    ⊢ CategoryTheory.IsPullback (c.ι.app { as := j }) (α.app { as := j }) i ((Cate …
  -/
  induction' j using Fin.inductionOn with j _
    /-
      case refine_3.zero
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsVanKampenColimit c₁
      t₂ : CategoryTheory.IsVanKampenColimit c₂
      inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      Hc : CategoryTheory.Limits.IsColimit c
      t₂' : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Categ …
      ⊢ CategoryTheory.IsPullback (c.ι.app { as := 0 }) (α.app { as := 0 }) i ((Cate …
    -/
  · exact t₂' ⟨WalkingPair.left⟩
    /-
      🎉 no goals
    -/
  · have t₁' := (@t₁ (Discrete.functor (fun j ↦ F.obj ⟨j.succ⟩)) (Cofan.mk _ _) (Discrete.natTrans
      fun i ↦ α.app _) (Sigma.desc (fun j ↦ α.app _ ≫ c₁.inj _)) ?_
      (NatTrans.equifibered_of_discrete _)).mp ⟨coproductIsCoproduct _⟩ ⟨j⟩
    /-
      case refine_3.succ.refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.IsVanKampenColimit c₁
      t₂ : CategoryTheory.IsVanKampenColimit c₂
      inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
      c : CategoryTheory.Limits.Cocone F
      α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
      i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
      e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
      hα : CategoryTheory.NatTrans.Equifibered α
      Hc : CategoryTheory.Limits.IsColimit c
      t₂' : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Categ …
      j : Fin n
      a✝ : CategoryTheory.IsPullback (c.ι.app { as := j.castSucc }) (α.app { as := j …
      t₁' : CategoryTheory.IsPullback ((CategoryTheory.Limits.Cofan.mk (CategoryTheo …
      ⊢ CategoryTheory.IsPullback (c.ι.app { as := j.succ }) (α.app { as := j.succ } …
    -/
    rotate_left
      /-
        case refine_3.succ.refine_1
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
        c : CategoryTheory.Limits.Cocone F
        α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        t₂' : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Categ …
        j : Fin n
        a✝ : CategoryTheory.IsPullback (c.ι.app { as := j.castSucc }) (α.app { as := j …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Discrete.natTrans fun …
      -/
    · ext ⟨j⟩
      /-
        case refine_3.succ.refine_1.w.h.mk
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
        c : CategoryTheory.Limits.Cocone F
        α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        t₂' : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Categ …
        j✝ : Fin n
        a✝ : CategoryTheory.IsPullback (c.ι.app { as := j✝.castSucc }) (α.app { as :=  …
        j : Fin n
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Discrete.natTrans fu …
      -/
      dsimp
      /-
        case refine_3.succ.refine_1.w.h.mk
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
        c : CategoryTheory.Limits.Cocone F
        α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        t₂' : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Categ …
        j✝ : Fin n
        a✝ : CategoryTheory.IsPullback (c.ι.app { as := j✝.castSucc }) (α.app { as :=  …
        j : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app { as := j.succ }) (c₁.ι.app {  …
      -/
      rw [colimit.ι_desc]
      /-
        case refine_3.succ.refine_1.w.h.mk
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.IsVanKampenColimit c₁
        t₂ : CategoryTheory.IsVanKampenColimit c₂
        inst✝¹ : ∀ {Z : C} (i : Quiver.Hom Z c₂.pt), CategoryTheory.Limits.HasPullback …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        F : CategoryTheory.Functor (CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))) C
        c : CategoryTheory.Limits.Cocone F
        α : Quiver.Hom F (CategoryTheory.Discrete.functor f)
        i : Quiver.Hom c.pt (CategoryTheory.extendCofan c₁ c₂).pt
        e : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.extendCofan c₁ c₂ …
        hα : CategoryTheory.NatTrans.Equifibered α
        Hc : CategoryTheory.Limits.IsColimit c
        t₂' : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Categ …
        j✝ : Fin n
        a✝ : CategoryTheory.IsPullback (c.ι.app { as := j✝.castSucc }) (α.app { as :=  …
        j : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app { as := j.succ }) (c₁.ι.app {  …
      -/
      rfl
      /-
        🎉 no goals
      -/
    simpa [Functor.const_obj_obj, Discrete.functor_obj, extendCofan_pt, extendCofan_ι_app,
      Fin.cases_succ, BinaryCofan.mk_pt, colimit.cocone_x, Cofan.mk_pt, Cofan.mk_ι_app,
      BinaryCofan.ι_app_right, BinaryCofan.mk_inr, colimit.ι_desc,
      Discrete.natTrans_app] using t₁'.paste_horiz (t₂' ⟨WalkingPair.right⟩)


theorem isPullback_of_cofan_isVanKampen [HasInitial C] {ι : Type*} {X : ι → C}
    {c : Cofan X} (hc : IsVanKampenColimit c) (i j : ι) [DecidableEq ι] :
    IsPullback (P := (if j = i then X i else ⊥_ C))
      (if h : j = i then eqToHom (if_pos h) else eqToHom (if_neg h) ≫ initial.to (X i))
      (if h : j = i then eqToHom ((if_pos h).trans (congr_arg X h.symm))
        else eqToHom (if_neg h) ≫ initial.to (X j))
      (Cofan.inj c i) (Cofan.inj c j) := by
  refine (hc (Cofan.mk (X i) (f := fun k ↦ if k = i then X i else ⊥_ C)
    (fun k ↦ if h : k = i then (eqToHom <| if_pos h) else (eqToHom <| if_neg h) ≫ initial.to _))
    (Discrete.natTrans (fun k ↦ if h : k.1 = i then (eqToHom <| (if_pos h).trans
      (congr_arg X h.symm)) else (eqToHom <| if_neg h) ≫ initial.to _))
    (c.inj i) ?_ (NatTrans.equifibered_of_discrete _)).mp ⟨?_⟩ ⟨j⟩
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasInitial C
      ι : Type u_3
      X : ι → C
      c : CategoryTheory.Limits.Cofan X
      hc : CategoryTheory.IsVanKampenColimit c
      i j : ι
      inst✝ : DecidableEq ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Discrete.natTrans fun …
    -/
  · ext ⟨k⟩
    simp only [Discrete.functor_obj, Functor.const_obj_obj, NatTrans.comp_app,
      Discrete.natTrans_app, Cofan.mk_pt, Cofan.mk_ι_app, Functor.const_map_app]
    /-
      case refine_1.w.h.mk
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasInitial C
      ι : Type u_3
      X : ι → C
      c : CategoryTheory.Limits.Cofan X
      hc : CategoryTheory.IsVanKampenColimit c
      i j : ι
      inst✝ : DecidableEq ι
      k : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq k i) (fun h => CategoryTheo …
    -/
    split
      /-
        case refine_1.w.h.mk.isTrue
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasInitial C
        ι : Type u_3
        X : ι → C
        c : CategoryTheory.Limits.Cofan X
        hc : CategoryTheory.IsVanKampenColimit c
        i j : ι
        inst✝ : DecidableEq ι
        k : ι
        h✝ : Eq k i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (c.ι.app { …
      -/
    · subst ‹k = i›; rfl
                     /-
                       🎉 no goals
                     -/
      /-
        case refine_1.w.h.mk.isFalse
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasInitial C
        ι : Type u_3
        X : ι → C
        c : CategoryTheory.Limits.Cofan X
        hc : CategoryTheory.IsVanKampenColimit c
        i j : ι
        inst✝ : DecidableEq ι
        k : ι
        h✝ : Not (Eq k i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasInitial C
      ι : Type u_3
      X : ι → C
      c : CategoryTheory.Limits.Cofan X
      hc : CategoryTheory.IsVanKampenColimit c
      i j : ι
      inst✝ : DecidableEq ι
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (X i) fun k  …
    -/
  · refine mkCofanColimit _ (fun t ↦ (eqToHom (if_pos rfl).symm) ≫ t.inj i) ?_ ?_
      /-
        case refine_2.refine_1
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasInitial C
        ι : Type u_3
        X : ι → C
        c : CategoryTheory.Limits.Cofan X
        hc : CategoryTheory.IsVanKampenColimit c
        i j : ι
        inst✝ : DecidableEq ι
        ⊢ ∀ (t : CategoryTheory.Limits.Cofan fun k => ite (Eq k i) (X i) (CategoryTheo …
      -/
    · intro t j
      /-
        case refine_2.refine_1
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasInitial C
        ι : Type u_3
        X : ι → C
        c : CategoryTheory.Limits.Cofan X
        hc : CategoryTheory.IsVanKampenColimit c
        i j✝ : ι
        inst✝ : DecidableEq ι
        t : CategoryTheory.Limits.Cofan fun k => ite (Eq k i) (X i) (CategoryTheory.Li …
        j : ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Cofan.mk (X i …
      -/
      simp only [Cofan.mk_pt, cofan_mk_inj]
      /-
        case refine_2.refine_1
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasInitial C
        ι : Type u_3
        X : ι → C
        c : CategoryTheory.Limits.Cofan X
        hc : CategoryTheory.IsVanKampenColimit c
        i j✝ : ι
        inst✝ : DecidableEq ι
        t : CategoryTheory.Limits.Cofan fun k => ite (Eq k i) (X i) (CategoryTheory.Li …
        j : ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq j i) (fun h => CategoryTheo …
      -/
      split
        /-
          case refine_2.refine_1.isTrue
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          ι : Type u_3
          X : ι → C
          c : CategoryTheory.Limits.Cofan X
          hc : CategoryTheory.IsVanKampenColimit c
          i j✝ : ι
          inst✝ : DecidableEq ι
          t : CategoryTheory.Limits.Cofan fun k => ite (Eq k i) (X i) (CategoryTheory.Li …
          j : ι
          h✝ : Eq j i
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
        -/
      · subst ‹j = i›; simp
                       /-
                         🎉 no goals
                       -/
        /-
          case refine_2.refine_1.isFalse
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          ι : Type u_3
          X : ι → C
          c : CategoryTheory.Limits.Cofan X
          hc : CategoryTheory.IsVanKampenColimit c
          i j✝ : ι
          inst✝ : DecidableEq ι
          t : CategoryTheory.Limits.Cofan fun k => ite (Eq k i) (X i) (CategoryTheory.Li …
          j : ι
          h✝ : Not (Eq j i)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · rw [Category.assoc, ← IsIso.eq_inv_comp]
        /-
          case refine_2.refine_1.isFalse
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasInitial C
          ι : Type u_3
          X : ι → C
          c : CategoryTheory.Limits.Cofan X
          hc : CategoryTheory.IsVanKampenColimit c
          i j✝ : ι
          inst✝ : DecidableEq ι
          t : CategoryTheory.Limits.Cofan fun k => ite (Eq k i) (X i) (CategoryTheory.Li …
          j : ι
          h✝ : Not (Eq j i)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.initial.to (X  …
        -/
        exact initialIsInitial.hom_ext _ _
        /-
          🎉 no goals
        -/
      /-
        case refine_2.refine_2
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasInitial C
        ι : Type u_3
        X : ι → C
        c : CategoryTheory.Limits.Cofan X
        hc : CategoryTheory.IsVanKampenColimit c
        i j : ι
        inst✝ : DecidableEq ι
        ⊢ ∀ (t : CategoryTheory.Limits.Cofan fun k => ite (Eq k i) (X i) (CategoryTheo …
      -/
    · intro t m hm
      /-
        case refine_2.refine_2
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasInitial C
        ι : Type u_3
        X : ι → C
        c : CategoryTheory.Limits.Cofan X
        hc : CategoryTheory.IsVanKampenColimit c
        i j : ι
        inst✝ : DecidableEq ι
        t : CategoryTheory.Limits.Cofan fun k => ite (Eq k i) (X i) (CategoryTheory.Li …
        m : Quiver.Hom (CategoryTheory.Limits.Cofan.mk (X i) fun k => dite (Eq k i) (f …
        hm : ∀ (j : ι), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits …
        ⊢ Eq m ((fun t => CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯ …
      -/
      simp [← hm i]
      /-
        🎉 no goals
      -/


theorem isPullback_initial_to_of_cofan_isVanKampen [HasInitial C] {ι : Type*} {F : Discrete ι ⥤ C}
    {c : Cocone F} (hc : IsVanKampenColimit c) (i j : Discrete ι) (hi : i ≠ j) :
    IsPullback (initial.to _) (initial.to _) (c.ι.app i) (c.ι.app j) := by
  classical
  let f : ι → C := F.obj ∘ Discrete.mk
  have : F = Discrete.functor f :=
    Functor.hext (fun i ↦ rfl) (by rintro ⟨i⟩ ⟨j⟩ ⟨⟨rfl : i = j⟩⟩; simp [f])
  clear_value f
  subst this
  have : ∀ i, Subsingleton (⊥_ C ⟶ (Discrete.functor f).obj i) := inferInstance
  convert isPullback_of_cofan_isVanKampen hc i.as j.as
  exact (if_neg (mt Discrete.ext hi.symm)).symm


theorem mono_of_cofan_isVanKampen [HasInitial C] {ι : Type*} {F : Discrete ι ⥤ C}
    {c : Cocone F} (hc : IsVanKampenColimit c) (i : Discrete ι) : Mono (c.ι.app i) := by
  classical
  let f : ι → C := F.obj ∘ Discrete.mk
  have : F = Discrete.functor f :=
    Functor.hext (fun i ↦ rfl) (by rintro ⟨i⟩ ⟨j⟩ ⟨⟨rfl : i = j⟩⟩; simp [f])
  clear_value f
  subst this
  refine PullbackCone.mono_of_isLimitMkIdId _ (IsPullback.isLimit ?_)
  nth_rw 1 [← Category.id_comp (c.ι.app i)]
  convert IsPullback.paste_vert _ (isPullback_of_cofan_isVanKampen hc i.as i.as)
  swap
  · exact (eqToHom (if_pos rfl).symm)
  · simp
  · exact IsPullback.of_vert_isIso ⟨by simp⟩


