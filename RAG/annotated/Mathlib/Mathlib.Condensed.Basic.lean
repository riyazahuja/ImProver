/--
`Condensed.{u} C` is the category of condensed objects in a category `C`, which are
defined as sheaves on `CompHaus.{u}` with respect to the coherent Grothendieck topology.
-/
def Condensed (C : Type w) [Category.{v} C] :=
  Sheaf (coherentTopology CompHaus.{u}) C


instance {C : Type w} [Category.{v} C] : Category (Condensed.{u} C) :=
  show Category (Sheaf _ _) from inferInstance


/--
Condensed sets (types) with the appropriate universe levels, i.e. `Type (u+1)`-valued
sheaves on `CompHaus.{u}`.
-/
abbrev CondensedSet := Condensed.{u} (Type (u+1))


@[simp]
lemma id_val (X : Condensed.{u} C) : (𝟙 X : X ⟶ X).val = 𝟙 _ := rfl


@[simp]
lemma comp_val {X Y Z : Condensed.{u} C} (f : X ⟶ Y) (g : Y ⟶ Z) : (f ≫ g).val = f.val ≫ g.val :=
  rfl


@[ext]
lemma hom_ext {X Y : Condensed.{u} C} (f g : X ⟶ Y) (h : ∀ S, f.val.app S = g.val.app S) :
    f = g := by
  /-
    C : Type w
    inst✝ : CategoryTheory.Category.{v, w} C
    X Y : Condensed C
    f g : Quiver.Hom X Y
    h : ∀ (S : Opposite CompHaus), Eq (f.val.app S) (g.val.app S)
    ⊢ Eq f g
  -/
  apply Sheaf.hom_ext
  /-
    case h
    C : Type w
    inst✝ : CategoryTheory.Category.{v, w} C
    X Y : Condensed C
    f g : Quiver.Hom X Y
    h : ∀ (S : Opposite CompHaus), Eq (f.val.app S) (g.val.app S)
    ⊢ Eq f.val g.val
  -/
  ext
  /-
    case h.w.h
    C : Type w
    inst✝ : CategoryTheory.Category.{v, w} C
    X Y : Condensed C
    f g : Quiver.Hom X Y
    h : ∀ (S : Opposite CompHaus), Eq (f.val.app S) (g.val.app S)
    x✝ : Opposite CompHaus
    ⊢ Eq (f.val.app x✝) (g.val.app x✝)
  -/
  exact h _
  /-
    🎉 no goals
  -/


@[simp]
lemma hom_naturality_apply {X Y : CondensedSet.{u}} (f : X ⟶ Y) {S T : CompHausᵒᵖ} (g : S ⟶ T)
    (x : X.val.obj S) : f.val.app T (X.val.map g x) = Y.val.map g (f.val.app S x) :=
  NatTrans.naturality_apply f.val g x


