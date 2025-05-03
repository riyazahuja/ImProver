/--
`LightCondensed.{u} C` is the category of light condensed objects in a category `C`, which are
defined as sheaves on `LightProfinite.{u}` with respect to the coherent Grothendieck topology.
-/
def LightCondensed (C : Type w) [Category.{v} C] :=
  Sheaf (coherentTopology LightProfinite.{u}) C


instance {C : Type w} [Category.{v} C] : Category (LightCondensed.{u} C) :=
  show Category (Sheaf _ _) from inferInstance


/--
Light condensed sets. Because `LightProfinite` is an essentially small category, we don't need the
same universe bump as in `CondensedSet`.
-/
abbrev LightCondSet := LightCondensed.{u} (Type u)


@[simp]
lemma id_val (X : LightCondensed.{u} C) : (𝟙 X : X ⟶ X).val = 𝟙 _ := rfl


@[simp]
lemma comp_val {X Y Z : LightCondensed.{u} C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).val = f.val ≫ g.val :=
  rfl


@[ext]
lemma hom_ext {X Y : LightCondensed.{u} C} (f g : X ⟶ Y) (h : ∀ S, f.val.app S = g.val.app S) :
    f = g := by
  /-
    C : Type w
    inst✝ : CategoryTheory.Category.{v, w} C
    X Y : LightCondensed C
    f g : Quiver.Hom X Y
    h : ∀ (S : Opposite LightProfinite), Eq (f.val.app S) (g.val.app S)
    ⊢ Eq f g
  -/
  apply Sheaf.hom_ext
  /-
    case h
    C : Type w
    inst✝ : CategoryTheory.Category.{v, w} C
    X Y : LightCondensed C
    f g : Quiver.Hom X Y
    h : ∀ (S : Opposite LightProfinite), Eq (f.val.app S) (g.val.app S)
    ⊢ Eq f.val g.val
  -/
  ext
  /-
    case h.w.h
    C : Type w
    inst✝ : CategoryTheory.Category.{v, w} C
    X Y : LightCondensed C
    f g : Quiver.Hom X Y
    h : ∀ (S : Opposite LightProfinite), Eq (f.val.app S) (g.val.app S)
    x✝ : Opposite LightProfinite
    ⊢ Eq (f.val.app x✝) (g.val.app x✝)
  -/
  exact h _
  /-
    🎉 no goals
  -/


@[simp]
lemma hom_naturality_apply {X Y : LightCondSet.{u}} (f : X ⟶ Y) {S T : LightProfiniteᵒᵖ}
    (g : S ⟶ T) (x : X.val.obj S) : f.val.app T (X.val.map g x) = Y.val.map g (f.val.app S x) :=
  NatTrans.naturality_apply f.val g x


