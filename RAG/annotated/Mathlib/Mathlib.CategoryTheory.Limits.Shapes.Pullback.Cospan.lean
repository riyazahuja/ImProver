/-- The type of objects for the diagram indexing a pullback, defined as a special case of
`WidePullbackShape`. -/
abbrev WalkingCospan : Type :=
  WidePullbackShape WalkingPair


/-- The left point of the walking cospan. -/
@[match_pattern]
abbrev WalkingCospan.left : WalkingCospan :=
  some WalkingPair.left


/-- The right point of the walking cospan. -/
@[match_pattern]
abbrev WalkingCospan.right : WalkingCospan :=
  some WalkingPair.right


/-- The central point of the walking cospan. -/
@[match_pattern]
abbrev WalkingCospan.one : WalkingCospan :=
  none


/-- The type of objects for the diagram indexing a pushout, defined as a special case of
`WidePushoutShape`.
-/
abbrev WalkingSpan : Type :=
  WidePushoutShape WalkingPair


/-- The left point of the walking span. -/
@[match_pattern]
abbrev WalkingSpan.left : WalkingSpan :=
  some WalkingPair.left


/-- The right point of the walking span. -/
@[match_pattern]
abbrev WalkingSpan.right : WalkingSpan :=
  some WalkingPair.right


/-- The central point of the walking span. -/
@[match_pattern]
abbrev WalkingSpan.zero : WalkingSpan :=
  none


/-- The type of arrows for the diagram indexing a pullback. -/
abbrev Hom : WalkingCospan → WalkingCospan → Type :=
  WidePullbackShape.Hom


/-- The left arrow of the walking cospan. -/
@[match_pattern]
abbrev Hom.inl : left ⟶ one :=
  WidePullbackShape.Hom.term _


/-- The right arrow of the walking cospan. -/
@[match_pattern]
abbrev Hom.inr : right ⟶ one :=
  WidePullbackShape.Hom.term _


/-- The identity arrows of the walking cospan. -/
@[match_pattern]
abbrev Hom.id (X : WalkingCospan) : X ⟶ X :=
  WidePullbackShape.Hom.id X


instance (X Y : WalkingCospan) : Subsingleton (X ⟶ Y) := by
  /-
    X Y : CategoryTheory.Limits.WalkingCospan
    ⊢ Subsingleton (Quiver.Hom X Y)
  -/
  constructor; intros; simp [eq_iff_true_of_subsingleton]
                       /-
                         🎉 no goals
                       -/


/-- The type of arrows for the diagram indexing a pushout. -/
abbrev Hom : WalkingSpan → WalkingSpan → Type :=
  WidePushoutShape.Hom


/-- The left arrow of the walking span. -/
@[match_pattern]
abbrev Hom.fst : zero ⟶ left :=
  WidePushoutShape.Hom.init _


/-- The right arrow of the walking span. -/
@[match_pattern]
abbrev Hom.snd : zero ⟶ right :=
  WidePushoutShape.Hom.init _


/-- The identity arrows of the walking span. -/
@[match_pattern]
abbrev Hom.id (X : WalkingSpan) : X ⟶ X :=
  WidePushoutShape.Hom.id X


instance (X Y : WalkingSpan) : Subsingleton (X ⟶ Y) := by
  /-
    X Y : CategoryTheory.Limits.WalkingSpan
    ⊢ Subsingleton (Quiver.Hom X Y)
  -/
  constructor; intros a b; simp [eq_iff_true_of_subsingleton]
                           /-
                             🎉 no goals
                           -/


/-- To construct an isomorphism of cones over the walking cospan,
it suffices to construct an isomorphism
of the cone points and check it commutes with the legs to `left` and `right`. -/
def WalkingCospan.ext {F : WalkingCospan ⥤ C} {s t : Cone F} (i : s.pt ≅ t.pt)
    (w₁ : s.π.app WalkingCospan.left = i.hom ≫ t.π.app WalkingCospan.left)
    (w₂ : s.π.app WalkingCospan.right = i.hom ≫ t.π.app WalkingCospan.right) : s ≅ t := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
    s t : CategoryTheory.Limits.Cone F
    i : CategoryTheory.Iso s.pt t.pt
    w₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.left) (CategoryTheory.Cat …
    w₂ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.right) (CategoryTheory.Ca …
    ⊢ CategoryTheory.Iso s t
  -/
  apply Cones.ext i _
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
    s t : CategoryTheory.Limits.Cone F
    i : CategoryTheory.Iso s.pt t.pt
    w₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.left) (CategoryTheory.Cat …
    w₂ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.right) (CategoryTheory.Ca …
    ⊢ ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (s.π.app j) (CategoryTheory. …
  -/
  rintro (⟨⟩ | ⟨⟨⟩⟩)
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
      s t : CategoryTheory.Limits.Cone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.left) (CategoryTheory.Cat …
      w₂ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.right) (CategoryTheory.Ca …
      ⊢ Eq (s.π.app Option.none) (CategoryTheory.CategoryStruct.comp i.hom (t.π.app  …
    -/
  · have h₁ := s.π.naturality WalkingCospan.Hom.inl
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
      s t : CategoryTheory.Limits.Cone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.left) (CategoryTheory.Cat …
      w₂ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.right) (CategoryTheory.Ca …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Ca …
      ⊢ Eq (s.π.app Option.none) (CategoryTheory.CategoryStruct.comp i.hom (t.π.app  …
    -/
    dsimp at h₁
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
      s t : CategoryTheory.Limits.Cone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.left) (CategoryTheory.Cat …
      w₂ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.right) (CategoryTheory.Ca …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id  …
      ⊢ Eq (s.π.app Option.none) (CategoryTheory.CategoryStruct.comp i.hom (t.π.app  …
    -/
    simp only [Category.id_comp] at h₁
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
      s t : CategoryTheory.Limits.Cone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.left) (CategoryTheory.Cat …
      w₂ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.right) (CategoryTheory.Ca …
      h₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.one) (CategoryTheory.Cate …
      ⊢ Eq (s.π.app Option.none) (CategoryTheory.CategoryStruct.comp i.hom (t.π.app  …
    -/
    have h₂ := t.π.naturality WalkingCospan.Hom.inl
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
      s t : CategoryTheory.Limits.Cone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.left) (CategoryTheory.Cat …
      w₂ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.right) (CategoryTheory.Ca …
      h₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.one) (CategoryTheory.Cate …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Ca …
      ⊢ Eq (s.π.app Option.none) (CategoryTheory.CategoryStruct.comp i.hom (t.π.app  …
    -/
    dsimp at h₂
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
      s t : CategoryTheory.Limits.Cone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.left) (CategoryTheory.Cat …
      w₂ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.right) (CategoryTheory.Ca …
      h₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.one) (CategoryTheory.Cate …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id  …
      ⊢ Eq (s.π.app Option.none) (CategoryTheory.CategoryStruct.comp i.hom (t.π.app  …
    -/
    simp only [Category.id_comp] at h₂
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
      s t : CategoryTheory.Limits.Cone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.left) (CategoryTheory.Cat …
      w₂ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.right) (CategoryTheory.Ca …
      h₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.one) (CategoryTheory.Cate …
      h₂ : Eq (t.π.app CategoryTheory.Limits.WalkingCospan.one) (CategoryTheory.Cate …
      ⊢ Eq (s.π.app Option.none) (CategoryTheory.CategoryStruct.comp i.hom (t.π.app  …
    -/
    simp_rw [h₂, ← Category.assoc, ← w₁, ← h₁]
    /-
      🎉 no goals
    -/
    /-
      case some.left
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
      s t : CategoryTheory.Limits.Cone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.left) (CategoryTheory.Cat …
      w₂ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.right) (CategoryTheory.Ca …
      ⊢ Eq (s.π.app (Option.some CategoryTheory.Limits.WalkingPair.left)) (CategoryT …
    -/
  · exact w₁
    /-
      🎉 no goals
    -/
    /-
      case some.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
      s t : CategoryTheory.Limits.Cone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.left) (CategoryTheory.Cat …
      w₂ : Eq (s.π.app CategoryTheory.Limits.WalkingCospan.right) (CategoryTheory.Ca …
      ⊢ Eq (s.π.app (Option.some CategoryTheory.Limits.WalkingPair.right)) (Category …
    -/
  · exact w₂
    /-
      🎉 no goals
    -/


/-- To construct an isomorphism of cocones over the walking span,
it suffices to construct an isomorphism
of the cocone points and check it commutes with the legs from `left` and `right`. -/
def WalkingSpan.ext {F : WalkingSpan ⥤ C} {s t : Cocone F} (i : s.pt ≅ t.pt)
    (w₁ : s.ι.app WalkingCospan.left ≫ i.hom = t.ι.app WalkingCospan.left)
    (w₂ : s.ι.app WalkingCospan.right ≫ i.hom = t.ι.app WalkingCospan.right) : s ≅ t := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
    s t : CategoryTheory.Limits.Cocone F
    i : CategoryTheory.Iso s.pt t.pt
    w₁ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
    w₂ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
    ⊢ CategoryTheory.Iso s t
  -/
  apply Cocones.ext i _
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
    s t : CategoryTheory.Limits.Cocone F
    i : CategoryTheory.Iso s.pt t.pt
    w₁ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
    w₂ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
    ⊢ ∀ (j : CategoryTheory.Limits.WalkingSpan), Eq (CategoryTheory.CategoryStruct …
  -/
  rintro (⟨⟩ | ⟨⟨⟩⟩)
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
      s t : CategoryTheory.Limits.Cocone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      w₂ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app Option.none) i.hom) (t.ι.app …
    -/
  · have h₁ := s.ι.naturality WalkingSpan.Hom.fst
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
      s t : CategoryTheory.Limits.Cocone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      w₂ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app Option.none) i.hom) (t.ι.app …
    -/
    dsimp at h₁
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
      s t : CategoryTheory.Limits.Cocone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      w₂ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app Option.none) i.hom) (t.ι.app …
    -/
    simp only [Category.comp_id] at h₁
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
      s t : CategoryTheory.Limits.Cocone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      w₂ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app Option.none) i.hom) (t.ι.app …
    -/
    have h₂ := t.ι.naturality WalkingSpan.Hom.fst
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
      s t : CategoryTheory.Limits.Cocone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      w₂ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app Option.none) i.hom) (t.ι.app …
    -/
    dsimp at h₂
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
      s t : CategoryTheory.Limits.Cocone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      w₂ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app Option.none) i.hom) (t.ι.app …
    -/
    simp only [Category.comp_id] at h₂
    /-
      case none
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
      s t : CategoryTheory.Limits.Cocone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      w₂ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app Option.none) i.hom) (t.ι.app …
    -/
    simp_rw [← h₁, Category.assoc, w₁, h₂]
    /-
      🎉 no goals
    -/
    /-
      case some.left
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
      s t : CategoryTheory.Limits.Cocone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      w₂ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app (Option.some CategoryTheory. …
    -/
  · exact w₁
    /-
      🎉 no goals
    -/
    /-
      case some.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
      s t : CategoryTheory.Limits.Cocone F
      i : CategoryTheory.Iso s.pt t.pt
      w₁ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      w₂ : Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Wal …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app (Option.some CategoryTheory. …
    -/
  · exact w₂
    /-
      🎉 no goals
    -/


/-- `cospan f g` is the functor from the walking cospan hitting `f` and `g`. -/
def cospan {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) : WalkingCospan ⥤ C :=
  WidePullbackShape.wideCospan Z (fun j => WalkingPair.casesOn j X Y) fun j =>
    WalkingPair.casesOn j f g


/-- `span f g` is the functor from the walking span hitting `f` and `g`. -/
def span {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) : WalkingSpan ⥤ C :=
  WidePushoutShape.wideSpan X (fun j => WalkingPair.casesOn j Y Z) fun j =>
    WalkingPair.casesOn j f g


@[simp]
theorem cospan_left {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) : (cospan f g).obj WalkingCospan.left = X :=
  rfl


@[simp]
theorem span_left {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) : (span f g).obj WalkingSpan.left = Y :=
  rfl


@[simp]
theorem cospan_right {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    (cospan f g).obj WalkingCospan.right = Y := rfl


@[simp]
theorem span_right {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) : (span f g).obj WalkingSpan.right = Z :=
  rfl


@[simp]
theorem cospan_one {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) : (cospan f g).obj WalkingCospan.one = Z :=
  rfl


@[simp]
theorem span_zero {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) : (span f g).obj WalkingSpan.zero = X :=
  rfl


@[simp]
theorem cospan_map_inl {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    (cospan f g).map WalkingCospan.Hom.inl = f := rfl


@[simp]
theorem span_map_fst {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) : (span f g).map WalkingSpan.Hom.fst = f :=
  rfl


@[simp]
theorem cospan_map_inr {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    (cospan f g).map WalkingCospan.Hom.inr = g := rfl


@[simp]
theorem span_map_snd {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) : (span f g).map WalkingSpan.Hom.snd = g :=
  rfl


theorem cospan_map_id {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) (w : WalkingCospan) :
    (cospan f g).map (WalkingCospan.Hom.id w) = 𝟙 _ := rfl


theorem span_map_id {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) (w : WalkingSpan) :
    (span f g).map (WalkingSpan.Hom.id w) = 𝟙 _ := rfl


/-- Every diagram indexing a pullback is naturally isomorphic (actually, equal) to a `cospan` -/
-- @[simps (config := { rhsMd := semireducible })]  Porting note: no semireducible
@[simps!]
def diagramIsoCospan (F : WalkingCospan ⥤ C) : F ≅ cospan (F.map inl) (F.map inr) :=
  NatIso.ofComponents
                        /-
                          C : Type u
                          inst✝ : CategoryTheory.Category.{v, u} C
                          F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
                          j : CategoryTheory.Limits.WalkingCospan
                          ⊢ Eq (F.obj j) ((CategoryTheory.Limits.cospan (F.map CategoryTheory.Limits.Wal …
                        -/
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  (fun j => eqToIso (by rcases j with (⟨⟩ | ⟨⟨⟩⟩) <;> rfl))
                                                      /-
                                                        🎉 no goals
                                                      -/
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingCospan C
        ⊢ ∀ {X Y : CategoryTheory.Limits.WalkingCospan} (f : Quiver.Hom X Y), Eq (Cate …
      -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  (by rintro (⟨⟩ | ⟨⟨⟩⟩) (⟨⟩ | ⟨⟨⟩⟩) f <;> cases f <;> dsimp <;> simp)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- Every diagram indexing a pushout is naturally isomorphic (actually, equal) to a `span` -/
-- @[simps (config := { rhsMd := semireducible })]  Porting note: no semireducible
@[simps!]
def diagramIsoSpan (F : WalkingSpan ⥤ C) : F ≅ span (F.map fst) (F.map snd) :=
  NatIso.ofComponents
                        /-
                          C : Type u
                          inst✝ : CategoryTheory.Category.{v, u} C
                          F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
                          j : CategoryTheory.Limits.WalkingSpan
                          ⊢ Eq (F.obj j) ((CategoryTheory.Limits.span (F.map CategoryTheory.Limits.Walki …
                        -/
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  (fun j => eqToIso (by rcases j with (⟨⟩ | ⟨⟨⟩⟩) <;> rfl))
                                                      /-
                                                        🎉 no goals
                                                      -/
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingSpan C
        ⊢ ∀ {X Y : CategoryTheory.Limits.WalkingSpan} (f : Quiver.Hom X Y), Eq (Catego …
      -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  (by rintro (⟨⟩ | ⟨⟨⟩⟩) (⟨⟩ | ⟨⟨⟩⟩) f <;> cases f <;> dsimp <;> simp)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- A functor applied to a cospan is a cospan. -/
def cospanCompIso (F : C ⥤ D) {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    cospan f g ⋙ F ≅ cospan (F.map f) (F.map g) :=
                          /-
                            C : Type u
                            inst✝¹ : CategoryTheory.Category.{v, u} C
                            D : Type u₂
                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                            F : CategoryTheory.Functor C D
                            X Y Z : C
                            f : Quiver.Hom X Z
                            g : Quiver.Hom Y Z
                            ⊢ (X_1 : CategoryTheory.Limits.WalkingCospan) → CategoryTheory.Iso (((Category …
                          -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  NatIso.ofComponents (by rintro (⟨⟩ | ⟨⟨⟩⟩) <;> exact Iso.refl _)
                                                 /-
                                                   🎉 no goals
                                                 -/
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          ⊢ ∀ {X_1 Y_1 : CategoryTheory.Limits.WalkingCospan} (f_1 : Quiver.Hom X_1 Y_1) …
        -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    (by rintro (⟨⟩ | ⟨⟨⟩⟩) (⟨⟩ | ⟨⟨⟩⟩) f <;> cases f <;> dsimp <;> simp)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem cospanCompIso_app_left : (cospanCompIso F f g).app WalkingCospan.left = Iso.refl _ := rfl


@[simp]
theorem cospanCompIso_app_right : (cospanCompIso F f g).app WalkingCospan.right = Iso.refl _ :=
  rfl


@[simp]
theorem cospanCompIso_app_one : (cospanCompIso F f g).app WalkingCospan.one = Iso.refl _ := rfl


@[simp]
theorem cospanCompIso_hom_app_left : (cospanCompIso F f g).hom.app WalkingCospan.left = 𝟙 _ :=
  rfl


@[simp]
theorem cospanCompIso_hom_app_right : (cospanCompIso F f g).hom.app WalkingCospan.right = 𝟙 _ :=
  rfl


@[simp]
theorem cospanCompIso_hom_app_one : (cospanCompIso F f g).hom.app WalkingCospan.one = 𝟙 _ := rfl


@[simp]
theorem cospanCompIso_inv_app_left : (cospanCompIso F f g).inv.app WalkingCospan.left = 𝟙 _ :=
  rfl


@[simp]
theorem cospanCompIso_inv_app_right : (cospanCompIso F f g).inv.app WalkingCospan.right = 𝟙 _ :=
  rfl


@[simp]
theorem cospanCompIso_inv_app_one : (cospanCompIso F f g).inv.app WalkingCospan.one = 𝟙 _ := rfl


/-- A functor applied to a span is a span. -/
def spanCompIso (F : C ⥤ D) {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) :
    span f g ⋙ F ≅ span (F.map f) (F.map g) :=
                          /-
                            C : Type u
                            inst✝¹ : CategoryTheory.Category.{v, u} C
                            D : Type u₂
                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                            F : CategoryTheory.Functor C D
                            X Y Z : C
                            f : Quiver.Hom X Y
                            g : Quiver.Hom X Z
                            ⊢ (X_1 : CategoryTheory.Limits.WalkingSpan) → CategoryTheory.Iso (((CategoryTh …
                          -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  NatIso.ofComponents (by rintro (⟨⟩ | ⟨⟨⟩⟩) <;> exact Iso.refl _)
                                                 /-
                                                   🎉 no goals
                                                 -/
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          X Y Z : C
          f : Quiver.Hom X Y
          g : Quiver.Hom X Z
          ⊢ ∀ {X_1 Y_1 : CategoryTheory.Limits.WalkingSpan} (f_1 : Quiver.Hom X_1 Y_1),  …
        -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    (by rintro (⟨⟩ | ⟨⟨⟩⟩) (⟨⟩ | ⟨⟨⟩⟩) f <;> cases f <;> dsimp <;> simp)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem spanCompIso_app_left : (spanCompIso F f g).app WalkingSpan.left = Iso.refl _ := rfl


@[simp]
theorem spanCompIso_app_right : (spanCompIso F f g).app WalkingSpan.right = Iso.refl _ := rfl


@[simp]
theorem spanCompIso_app_zero : (spanCompIso F f g).app WalkingSpan.zero = Iso.refl _ := rfl


@[simp]
theorem spanCompIso_hom_app_left : (spanCompIso F f g).hom.app WalkingSpan.left = 𝟙 _ := rfl


@[simp]
theorem spanCompIso_hom_app_right : (spanCompIso F f g).hom.app WalkingSpan.right = 𝟙 _ := rfl


@[simp]
theorem spanCompIso_hom_app_zero : (spanCompIso F f g).hom.app WalkingSpan.zero = 𝟙 _ := rfl


@[simp]
theorem spanCompIso_inv_app_left : (spanCompIso F f g).inv.app WalkingSpan.left = 𝟙 _ := rfl


@[simp]
theorem spanCompIso_inv_app_right : (spanCompIso F f g).inv.app WalkingSpan.right = 𝟙 _ := rfl


@[simp]
theorem spanCompIso_inv_app_zero : (spanCompIso F f g).inv.app WalkingSpan.zero = 𝟙 _ := rfl


/-- Construct an isomorphism of cospans from components. -/
def cospanExt (wf : iX.hom ≫ f' = f ≫ iZ.hom) (wg : iY.hom ≫ g' = g ≫ iZ.hom) :
    cospan f g ≅ cospan f' g' :=
  NatIso.ofComponents
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X Y Z X' Y' Z' : C
          iX : CategoryTheory.Iso X X'
          iY : CategoryTheory.Iso Y Y'
          iZ : CategoryTheory.Iso Z Z'
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          f' : Quiver.Hom X' Z'
          g' : Quiver.Hom Y' Z'
          wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
          wg : Eq (CategoryTheory.CategoryStruct.comp iY.hom g') (CategoryTheory.Categor …
          ⊢ (X_1 : CategoryTheory.Limits.WalkingCospan) → CategoryTheory.Iso ((CategoryT …
        -/
    (by rintro (⟨⟩ | ⟨⟨⟩⟩); exacts [iZ, iX, iY])
                            /-
                              🎉 no goals
                            -/
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X Y Z X' Y' Z' : C
          iX : CategoryTheory.Iso X X'
          iY : CategoryTheory.Iso Y Y'
          iZ : CategoryTheory.Iso Z Z'
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          f' : Quiver.Hom X' Z'
          g' : Quiver.Hom Y' Z'
          wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
          wg : Eq (CategoryTheory.CategoryStruct.comp iY.hom g') (CategoryTheory.Categor …
          ⊢ ∀ {X_1 Y_1 : CategoryTheory.Limits.WalkingCospan} (f_1 : Quiver.Hom X_1 Y_1) …
        -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    (by rintro (⟨⟩ | ⟨⟨⟩⟩) (⟨⟩ | ⟨⟨⟩⟩) f <;> cases f <;> dsimp <;> simp [wf, wg])
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem cospanExt_app_left : (cospanExt iX iY iZ wf wg).app WalkingCospan.left = iX := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    f' : Quiver.Hom X' Z'
    g' : Quiver.Hom Y' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iY.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.cospanExt iX iY iZ wf wg).app CategoryTheory.Limi …
  -/
  dsimp [cospanExt]
  /-
    🎉 no goals
  -/


@[simp]
theorem cospanExt_app_right : (cospanExt iX iY iZ wf wg).app WalkingCospan.right = iY := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    f' : Quiver.Hom X' Z'
    g' : Quiver.Hom Y' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iY.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.cospanExt iX iY iZ wf wg).app CategoryTheory.Limi …
  -/
  dsimp [cospanExt]
  /-
    🎉 no goals
  -/


@[simp]
theorem cospanExt_app_one : (cospanExt iX iY iZ wf wg).app WalkingCospan.one = iZ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    f' : Quiver.Hom X' Z'
    g' : Quiver.Hom Y' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iY.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.cospanExt iX iY iZ wf wg).app CategoryTheory.Limi …
  -/
  dsimp [cospanExt]
  /-
    🎉 no goals
  -/


@[simp]
theorem cospanExt_hom_app_left :
                                                                         /-
                                                                           C : Type u
                                                                           inst✝ : CategoryTheory.Category.{v, u} C
                                                                           X Y Z X' Y' Z' : C
                                                                           iX : CategoryTheory.Iso X X'
                                                                           iY : CategoryTheory.Iso Y Y'
                                                                           iZ : CategoryTheory.Iso Z Z'
                                                                           f : Quiver.Hom X Z
                                                                           g : Quiver.Hom Y Z
                                                                           f' : Quiver.Hom X' Z'
                                                                           g' : Quiver.Hom Y' Z'
                                                                           wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
                                                                           wg : Eq (CategoryTheory.CategoryStruct.comp iY.hom g') (CategoryTheory.Categor …
                                                                           ⊢ Eq ((CategoryTheory.Limits.cospanExt iX iY iZ wf wg).hom.app CategoryTheory. …
                                                                         -/
    (cospanExt iX iY iZ wf wg).hom.app WalkingCospan.left = iX.hom := by dsimp [cospanExt]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
theorem cospanExt_hom_app_right :
                                                                          /-
                                                                            C : Type u
                                                                            inst✝ : CategoryTheory.Category.{v, u} C
                                                                            X Y Z X' Y' Z' : C
                                                                            iX : CategoryTheory.Iso X X'
                                                                            iY : CategoryTheory.Iso Y Y'
                                                                            iZ : CategoryTheory.Iso Z Z'
                                                                            f : Quiver.Hom X Z
                                                                            g : Quiver.Hom Y Z
                                                                            f' : Quiver.Hom X' Z'
                                                                            g' : Quiver.Hom Y' Z'
                                                                            wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
                                                                            wg : Eq (CategoryTheory.CategoryStruct.comp iY.hom g') (CategoryTheory.Categor …
                                                                            ⊢ Eq ((CategoryTheory.Limits.cospanExt iX iY iZ wf wg).hom.app CategoryTheory. …
                                                                          -/
    (cospanExt iX iY iZ wf wg).hom.app WalkingCospan.right = iY.hom := by dsimp [cospanExt]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
theorem cospanExt_hom_app_one : (cospanExt iX iY iZ wf wg).hom.app WalkingCospan.one = iZ.hom := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    f' : Quiver.Hom X' Z'
    g' : Quiver.Hom Y' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iY.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.cospanExt iX iY iZ wf wg).hom.app CategoryTheory. …
  -/
  dsimp [cospanExt]
  /-
    🎉 no goals
  -/


@[simp]
theorem cospanExt_inv_app_left :
                                                                         /-
                                                                           C : Type u
                                                                           inst✝ : CategoryTheory.Category.{v, u} C
                                                                           X Y Z X' Y' Z' : C
                                                                           iX : CategoryTheory.Iso X X'
                                                                           iY : CategoryTheory.Iso Y Y'
                                                                           iZ : CategoryTheory.Iso Z Z'
                                                                           f : Quiver.Hom X Z
                                                                           g : Quiver.Hom Y Z
                                                                           f' : Quiver.Hom X' Z'
                                                                           g' : Quiver.Hom Y' Z'
                                                                           wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
                                                                           wg : Eq (CategoryTheory.CategoryStruct.comp iY.hom g') (CategoryTheory.Categor …
                                                                           ⊢ Eq ((CategoryTheory.Limits.cospanExt iX iY iZ wf wg).inv.app CategoryTheory. …
                                                                         -/
    (cospanExt iX iY iZ wf wg).inv.app WalkingCospan.left = iX.inv := by dsimp [cospanExt]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
theorem cospanExt_inv_app_right :
                                                                          /-
                                                                            C : Type u
                                                                            inst✝ : CategoryTheory.Category.{v, u} C
                                                                            X Y Z X' Y' Z' : C
                                                                            iX : CategoryTheory.Iso X X'
                                                                            iY : CategoryTheory.Iso Y Y'
                                                                            iZ : CategoryTheory.Iso Z Z'
                                                                            f : Quiver.Hom X Z
                                                                            g : Quiver.Hom Y Z
                                                                            f' : Quiver.Hom X' Z'
                                                                            g' : Quiver.Hom Y' Z'
                                                                            wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
                                                                            wg : Eq (CategoryTheory.CategoryStruct.comp iY.hom g') (CategoryTheory.Categor …
                                                                            ⊢ Eq ((CategoryTheory.Limits.cospanExt iX iY iZ wf wg).inv.app CategoryTheory. …
                                                                          -/
    (cospanExt iX iY iZ wf wg).inv.app WalkingCospan.right = iY.inv := by dsimp [cospanExt]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
theorem cospanExt_inv_app_one : (cospanExt iX iY iZ wf wg).inv.app WalkingCospan.one = iZ.inv := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    f' : Quiver.Hom X' Z'
    g' : Quiver.Hom Y' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iY.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.cospanExt iX iY iZ wf wg).inv.app CategoryTheory. …
  -/
  dsimp [cospanExt]
  /-
    🎉 no goals
  -/


/-- Construct an isomorphism of spans from components. -/
def spanExt (wf : iX.hom ≫ f' = f ≫ iY.hom) (wg : iX.hom ≫ g' = g ≫ iZ.hom) :
    span f g ≅ span f' g' :=
                          /-
                            C : Type u
                            inst✝¹ : CategoryTheory.Category.{v, u} C
                            D : Type u₂
                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                            X Y Z X' Y' Z' : C
                            iX : CategoryTheory.Iso X X'
                            iY : CategoryTheory.Iso Y Y'
                            iZ : CategoryTheory.Iso Z Z'
                            f : Quiver.Hom X Y
                            g : Quiver.Hom X Z
                            f' : Quiver.Hom X' Y'
                            g' : Quiver.Hom X' Z'
                            wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
                            wg : Eq (CategoryTheory.CategoryStruct.comp iX.hom g') (CategoryTheory.Categor …
                            ⊢ (X_1 : CategoryTheory.Limits.WalkingSpan) → CategoryTheory.Iso ((CategoryThe …
                          -/
  NatIso.ofComponents (by rintro (⟨⟩ | ⟨⟨⟩⟩); exacts [iX, iY, iZ])
                                              /-
                                                🎉 no goals
                                              -/
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X Y Z X' Y' Z' : C
          iX : CategoryTheory.Iso X X'
          iY : CategoryTheory.Iso Y Y'
          iZ : CategoryTheory.Iso Z Z'
          f : Quiver.Hom X Y
          g : Quiver.Hom X Z
          f' : Quiver.Hom X' Y'
          g' : Quiver.Hom X' Z'
          wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
          wg : Eq (CategoryTheory.CategoryStruct.comp iX.hom g') (CategoryTheory.Categor …
          ⊢ ∀ {X_1 Y_1 : CategoryTheory.Limits.WalkingSpan} (f_1 : Quiver.Hom X_1 Y_1),  …
        -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    (by rintro (⟨⟩ | ⟨⟨⟩⟩) (⟨⟩ | ⟨⟨⟩⟩) f <;> cases f <;> dsimp <;> simp [wf, wg])
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem spanExt_app_left : (spanExt iX iY iZ wf wg).app WalkingSpan.left = iY := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    f' : Quiver.Hom X' Y'
    g' : Quiver.Hom X' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iX.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.spanExt iX iY iZ wf wg).app CategoryTheory.Limits …
  -/
  dsimp [spanExt]
  /-
    🎉 no goals
  -/


@[simp]
theorem spanExt_app_right : (spanExt iX iY iZ wf wg).app WalkingSpan.right = iZ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    f' : Quiver.Hom X' Y'
    g' : Quiver.Hom X' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iX.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.spanExt iX iY iZ wf wg).app CategoryTheory.Limits …
  -/
  dsimp [spanExt]
  /-
    🎉 no goals
  -/


@[simp]
theorem spanExt_app_one : (spanExt iX iY iZ wf wg).app WalkingSpan.zero = iX := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    f' : Quiver.Hom X' Y'
    g' : Quiver.Hom X' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iX.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.spanExt iX iY iZ wf wg).app CategoryTheory.Limits …
  -/
  dsimp [spanExt]
  /-
    🎉 no goals
  -/


@[simp]
theorem spanExt_hom_app_left : (spanExt iX iY iZ wf wg).hom.app WalkingSpan.left = iY.hom := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    f' : Quiver.Hom X' Y'
    g' : Quiver.Hom X' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iX.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.spanExt iX iY iZ wf wg).hom.app CategoryTheory.Li …
  -/
  dsimp [spanExt]
  /-
    🎉 no goals
  -/


@[simp]
theorem spanExt_hom_app_right : (spanExt iX iY iZ wf wg).hom.app WalkingSpan.right = iZ.hom := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    f' : Quiver.Hom X' Y'
    g' : Quiver.Hom X' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iX.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.spanExt iX iY iZ wf wg).hom.app CategoryTheory.Li …
  -/
  dsimp [spanExt]
  /-
    🎉 no goals
  -/


@[simp]
theorem spanExt_hom_app_zero : (spanExt iX iY iZ wf wg).hom.app WalkingSpan.zero = iX.hom := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    f' : Quiver.Hom X' Y'
    g' : Quiver.Hom X' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iX.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.spanExt iX iY iZ wf wg).hom.app CategoryTheory.Li …
  -/
  dsimp [spanExt]
  /-
    🎉 no goals
  -/


@[simp]
theorem spanExt_inv_app_left : (spanExt iX iY iZ wf wg).inv.app WalkingSpan.left = iY.inv := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    f' : Quiver.Hom X' Y'
    g' : Quiver.Hom X' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iX.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.spanExt iX iY iZ wf wg).inv.app CategoryTheory.Li …
  -/
  dsimp [spanExt]
  /-
    🎉 no goals
  -/


@[simp]
theorem spanExt_inv_app_right : (spanExt iX iY iZ wf wg).inv.app WalkingSpan.right = iZ.inv := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    f' : Quiver.Hom X' Y'
    g' : Quiver.Hom X' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iX.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.spanExt iX iY iZ wf wg).inv.app CategoryTheory.Li …
  -/
  dsimp [spanExt]
  /-
    🎉 no goals
  -/


@[simp]
theorem spanExt_inv_app_zero : (spanExt iX iY iZ wf wg).inv.app WalkingSpan.zero = iX.inv := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z X' Y' Z' : C
    iX : CategoryTheory.Iso X X'
    iY : CategoryTheory.Iso Y Y'
    iZ : CategoryTheory.Iso Z Z'
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    f' : Quiver.Hom X' Y'
    g' : Quiver.Hom X' Z'
    wf : Eq (CategoryTheory.CategoryStruct.comp iX.hom f') (CategoryTheory.Categor …
    wg : Eq (CategoryTheory.CategoryStruct.comp iX.hom g') (CategoryTheory.Categor …
    ⊢ Eq ((CategoryTheory.Limits.spanExt iX iY iZ wf wg).inv.app CategoryTheory.Li …
  -/
  dsimp [spanExt]
  /-
    🎉 no goals
  -/


