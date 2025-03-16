/-- A morphism `φ : a ⟶ b` in `𝒳` lying over `f : R ⟶ S` in `𝒮` is cocartesian if for all
morphisms `φ' : a ⟶ b'`, also lying over `f`, there exists a unique morphism `χ : b ⟶ b'` lifting
`𝟙 S` such that `φ' = φ ≫ χ`. -/
class IsCocartesian extends IsHomLift p f φ : Prop where
  universal_property {b' : 𝒳} (φ' : a ⟶ b') [IsHomLift p f φ'] :
      ∃! χ : b ⟶ b', IsHomLift p (𝟙 S) χ ∧ φ ≫ χ = φ'


/-- A morphism `φ : a ⟶ b` in `𝒳` lying over `f : R ⟶ S` in `𝒮` is strongly cocartesian if for
all morphisms `φ' : a ⟶ b'` and all diagrams of the form
```
a --φ--> b        b'
|        |        |
v        v        v
R --f--> S --g--> S'
```
such that `φ'` lifts `f ≫ g`, there exists a lift `χ` of `g` such that `φ' = χ ≫ φ`.

See <https://stacks.math.columbia.edu/tag/02XK>. -/
class IsStronglyCocartesian extends IsHomLift p f φ : Prop where
  universal_property' {b' : 𝒳} (g : S ⟶ p.obj b') (φ' : a ⟶ b') [IsHomLift p (f ≫ g) φ'] :
      ∃! χ : b ⟶ b', IsHomLift p g χ ∧ φ ≫ χ = φ'


/-- Given a cocartesian morphism `φ : a ⟶ b` lying over `f : R ⟶ S` in `𝒳`, and another morphism
`φ' : a ⟶ b'` which also lifts `f`, then `IsCocartesian.map f φ φ'` is the morphism `b ⟶ b'` lying
over `𝟙 S` obtained from the universal property of `φ`. -/
protected noncomputable def map : b ⟶ b' :=
  Classical.choose <| IsCocartesian.universal_property (p:=p) (f:=f) (φ:=φ) φ'


instance map_isHomLift : IsHomLift p (𝟙 S) (IsCocartesian.map p f φ φ') :=
  (Classical.choose_spec <| IsCocartesian.universal_property (p:=p) (f:=f) (φ:=φ) φ').1.1


@[reassoc (attr := simp)]
lemma fac : φ ≫ IsCocartesian.map p f φ φ' = φ' :=
  (Classical.choose_spec <| IsCocartesian.universal_property (p:=p) (f:=f) (φ:=φ) φ').1.2


/-- Given a cocartesian morphism `φ : a ⟶ b` lying over `f : R ⟶ S` in `𝒳`, and another morphism
`φ' : a ⟶ b'` which also lifts `f`. Then any morphism `ψ : b ⟶ b'` lifting `𝟙 S` such that
`g ≫ ψ = φ'` must equal the map induced by the universal property of `φ`. -/
lemma map_uniq (ψ : b ⟶ b') [IsHomLift p (𝟙 S) ψ] (hψ : φ ≫ ψ = φ') :
    ψ = IsCocartesian.map p f φ φ' :=
  (Classical.choose_spec <| IsCocartesian.universal_property (p:=p) (f:=f) (φ:=φ) φ').2
    ψ ⟨inferInstance, hψ⟩


/-- Given a cocartesian morphism `φ : a ⟶ b` lying over `f : R ⟶ S` in `𝒳`, and two morphisms
`ψ ψ' : b ⟶ b'` lifting `𝟙 S` such that `φ ≫ ψ = φ ≫ ψ'`. Then we must have `ψ = ψ'`. -/
protected lemma ext (φ : a ⟶ b) [IsCocartesian p f φ] {b' : 𝒳} (ψ ψ' : b ⟶ b')
    [IsHomLift p (𝟙 S) ψ] [IsHomLift p (𝟙 S) ψ'] (h : φ ≫ ψ = φ ≫ ψ') : ψ = ψ' := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝² : p.IsCocartesian f φ
    b' : 𝒳
    ψ ψ' : Quiver.Hom b b'
    inst✝¹ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) ψ
    inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) ψ'
    h : Eq (CategoryTheory.CategoryStruct.comp φ ψ) (CategoryTheory.CategoryStruct …
    ⊢ Eq ψ ψ'
  -/
  rw [map_uniq p f φ (φ ≫ ψ) ψ rfl, map_uniq p f φ (φ ≫ ψ) ψ' h.symm]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_self : IsCocartesian.map p f φ φ = 𝟙 b := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝ : p.IsCocartesian f φ
    ⊢ Eq (CategoryTheory.Functor.IsCocartesian.map p f φ φ) (CategoryTheory.Catego …
  -/
  subst_hom_lift p f φ; symm
  /-
    case map
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a✝ b✝ : 𝒳
    φ : Quiver.Hom a✝ b✝
    R S : 𝒮
    a b : 𝒳
    inst✝ : p.IsCocartesian (p.map φ) φ
    ⊢ Eq (CategoryTheory.CategoryStruct.id b✝) (CategoryTheory.Functor.IsCocartesi …
  -/
  apply map_uniq
  /-
    case map.hψ
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a✝ b✝ : 𝒳
    φ : Quiver.Hom a✝ b✝
    R S : 𝒮
    a b : 𝒳
    inst✝ : p.IsCocartesian (p.map φ) φ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.id b …
  -/
  simp only [comp_id]
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism between the codomains of two cocartesian morphisms
lying over the same object. -/
noncomputable def codomainUniqueUpToIso {b' : 𝒳} (φ' : a ⟶ b') [IsCocartesian p f φ'] :
    b ≅ b' where
  hom := IsCocartesian.map p f φ φ'
  inv := IsCocartesian.map p f φ' φ
  hom_inv_id := by
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      b' : 𝒳
      φ' : Quiver.Hom a b'
      inst✝ : p.IsCocartesian f φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCocartesian …
    -/
    subst_hom_lift p f φ
    /-
      case map
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      a✝ b✝ : 𝒳
      φ : Quiver.Hom a✝ b✝
      b' : 𝒳
      φ' : Quiver.Hom a✝ b'
      R S : 𝒮
      a b : 𝒳
      inst✝¹ : p.IsCocartesian (p.map φ) φ
      inst✝ : p.IsCocartesian (p.map φ) φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCocartesian …
    -/
    apply IsCocartesian.ext p (p.map φ) φ
    /-
      case map.h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      a✝ b✝ : 𝒳
      φ : Quiver.Hom a✝ b✝
      b' : 𝒳
      φ' : Quiver.Hom a✝ b'
      R S : 𝒮
      a b : 𝒳
      inst✝¹ : p.IsCocartesian (p.map φ) φ
      inst✝ : p.IsCocartesian (p.map φ) φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
    -/
    simp only [fac_assoc, fac, comp_id]
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      b' : 𝒳
      φ' : Quiver.Hom a b'
      inst✝ : p.IsCocartesian f φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCocartesian …
    -/
    subst_hom_lift p f φ'
    /-
      case map
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      a✝ b✝ : 𝒳
      φ : Quiver.Hom a✝ b✝
      b' : 𝒳
      φ' : Quiver.Hom a✝ b'
      R S : 𝒮
      a b : 𝒳
      inst✝¹ : p.IsCocartesian (p.map φ') φ
      inst✝ : p.IsCocartesian (p.map φ') φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCocartesian …
    -/
    apply IsCocartesian.ext p (p.map φ') φ'
    /-
      case map.h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      a✝ b✝ : 𝒳
      φ : Quiver.Hom a✝ b✝
      b' : 𝒳
      φ' : Quiver.Hom a✝ b'
      R S : 𝒮
      a b : 𝒳
      inst✝¹ : p.IsCocartesian (p.map φ') φ
      inst✝ : p.IsCocartesian (p.map φ') φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ' (CategoryTheory.CategoryStruct.com …
    -/
    simp only [fac_assoc, fac, comp_id]
    /-
      🎉 no goals
    -/


/-- Postcomposing a cocartesian morphism with an isomorphism lifting the identity is cocartesian. -/
instance of_comp_iso {b' : 𝒳} (φ' : b ≅ b') [IsHomLift p (𝟙 S) φ'.hom] :
    IsCocartesian p f (φ ≫ φ'.hom) where
  universal_property := by
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      ⊢ ∀ {b'_1 : 𝒳} (φ'_1 : Quiver.Hom a b'_1) [inst : p.IsHomLift f φ'_1], ExistsU …
    -/
    intro c ψ hψ
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom a c
      hψ : p.IsHomLift f ψ
      ⊢ ExistsUnique fun χ => And (p.IsHomLift (CategoryTheory.CategoryStruct.id S)  …
    -/
    use φ'.inv ≫ IsCocartesian.map p f φ ψ
    /-
      case h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom a c
      hψ : p.IsHomLift f ψ
      ⊢ And ((fun χ => And (p.IsHomLift (CategoryTheory.CategoryStruct.id S) χ) (Eq  …
    -/
    refine ⟨⟨inferInstance, by simp⟩, ?_⟩
    /-
      case h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom a c
      hψ : p.IsHomLift f ψ
      ⊢ ∀ (y : Quiver.Hom b' c), (fun χ => And (p.IsHomLift (CategoryTheory.Category …
    -/
    rintro τ ⟨hτ₁, hτ₂⟩
    /-
      case h.intro
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom a c
      hψ : p.IsHomLift f ψ
      τ : Quiver.Hom b' c
      hτ₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) τ
      hτ₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
      ⊢ Eq τ (CategoryTheory.CategoryStruct.comp φ'.inv (CategoryTheory.Functor.IsCo …
    -/
    rw [Iso.eq_inv_comp]
    /-
      case h.intro
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom a c
      hψ : p.IsHomLift f ψ
      τ : Quiver.Hom b' c
      hτ₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) τ
      hτ₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ'.hom τ) (CategoryTheory.Functor.IsC …
    -/
    apply map_uniq
    /-
      case h.intro.hψ
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom a c
      hψ : p.IsHomLift f ψ
      τ : Quiver.Hom b' c
      hτ₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) τ
      hτ₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
    -/
    exact ((assoc φ _ _) ▸ hτ₂)
    /-
      🎉 no goals
    -/


/-- Precomposing a cocartesian morphism with an isomorphism lifting the identity is cocartesian. -/
instance of_iso_comp {a' : 𝒳} (φ' : a' ≅ a) [IsHomLift p (𝟙 R) φ'.hom] :
    IsCocartesian p f (φ'.hom ≫ φ) where
  universal_property := by
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      ⊢ ∀ {b' : 𝒳} (φ'_1 : Quiver.Hom a' b') [inst : p.IsHomLift f φ'_1], ExistsUniq …
    -/
    intro c ψ hψ
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom a' c
      hψ : p.IsHomLift f ψ
      ⊢ ExistsUnique fun χ => And (p.IsHomLift (CategoryTheory.CategoryStruct.id S)  …
    -/
    use IsCocartesian.map p f φ (φ'.inv ≫ ψ)
    /-
      case h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom a' c
      hψ : p.IsHomLift f ψ
      ⊢ And ((fun χ => And (p.IsHomLift (CategoryTheory.CategoryStruct.id S) χ) (Eq  …
    -/
    refine ⟨⟨inferInstance, by simp⟩, ?_⟩
    /-
      case h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom a' c
      hψ : p.IsHomLift f ψ
      ⊢ ∀ (y : Quiver.Hom b c), (fun χ => And (p.IsHomLift (CategoryTheory.CategoryS …
    -/
    rintro τ ⟨hτ₁, hτ₂⟩
    /-
      case h.intro
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom a' c
      hψ : p.IsHomLift f ψ
      τ : Quiver.Hom b c
      hτ₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) τ
      hτ₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
      ⊢ Eq τ (CategoryTheory.Functor.IsCocartesian.map p f φ (CategoryTheory.Categor …
    -/
    apply map_uniq
    /-
      case h.intro.hψ
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : Quiver.Hom a b
      inst✝¹ : p.IsCocartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom a' c
      hψ : p.IsHomLift f ψ
      τ : Quiver.Hom b c
      hτ₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) τ
      hτ₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ τ) (CategoryTheory.CategoryStruct.c …
    -/
    simp only [Iso.eq_inv_comp, ← assoc, hτ₂]
    /-
      🎉 no goals
    -/


/-- The universal property of a strongly cocartesian morphism.

This lemma is more flexible with respect to non-definitional equalities than the field
`universal_property'` of `IsStronglyCocartesian`. -/
lemma universal_property {S' : 𝒮} {b' : 𝒳} (g : S ⟶ S') (f' : R ⟶ S') (hf' : f' = f ≫ g)
    (φ' : a ⟶ b') [IsHomLift p f' φ'] : ∃! χ : b ⟶ b', IsHomLift p g χ ∧ φ ≫ χ = φ' := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCocartesian f φ
    S' : 𝒮
    b' : 𝒳
    g : Quiver.Hom S S'
    f' : Quiver.Hom R S'
    hf' : Eq f' (CategoryTheory.CategoryStruct.comp f g)
    φ' : Quiver.Hom a b'
    inst✝ : p.IsHomLift f' φ'
    ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
  -/
  subst_hom_lift p f' φ'; clear a b R S
  /-
    case map
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    S : 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    b' : 𝒳
    φ' : Quiver.Hom a b'
    f : Quiver.Hom (p.obj a) S
    inst✝¹ : p.IsStronglyCocartesian f φ
    g : Quiver.Hom S (p.obj b')
    hf' : Eq (p.map φ') (CategoryTheory.CategoryStruct.comp f g)
    inst✝ : p.IsHomLift (p.map φ') φ'
    ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
  -/
  have : p.IsHomLift (f ≫ g) φ' := (hf' ▸ inferInstance)
  /-
    case map
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    S : 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    b' : 𝒳
    φ' : Quiver.Hom a b'
    f : Quiver.Hom (p.obj a) S
    inst✝¹ : p.IsStronglyCocartesian f φ
    g : Quiver.Hom S (p.obj b')
    hf' : Eq (p.map φ') (CategoryTheory.CategoryStruct.comp f g)
    inst✝ : p.IsHomLift (p.map φ') φ'
    this : p.IsHomLift (CategoryTheory.CategoryStruct.comp f g) φ'
    ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
  -/
  apply IsStronglyCocartesian.universal_property' f
  /-
    🎉 no goals
  -/


instance isCocartesian_of_isStronglyCocartesian [p.IsStronglyCocartesian f φ] :
    p.IsCocartesian f φ where
  universal_property := fun φ' => universal_property p f φ (𝟙 S) f (comp_id f).symm φ'


/-- Given a diagram
```
a --φ--> b        b'
|        |        |
v        v        v
R --f--> S --g--> S'
```
such that `φ` is strongly cocartesian, and a morphism `φ' : a ⟶ b'`. Then `map` is the map
`b ⟶ b'` lying over `g` obtained from the universal property of `φ`. -/
noncomputable def map : b ⟶ b' :=
  Classical.choose <| universal_property p f φ _ _ hf' φ'


instance map_isHomLift : IsHomLift p g (map p f φ hf' φ') :=
  (Classical.choose_spec <| universal_property p f φ _ _ hf' φ').1.1


@[reassoc (attr := simp)]
lemma fac : φ ≫ (map p f φ hf' φ') = φ' :=
  (Classical.choose_spec <| universal_property p f φ _ _ hf' φ').1.2



/-- Given a diagram
```
a --φ--> b        b'
|        |        |
v        v        v
R --f--> S --g--> S'
```
such that `φ` is strongly cocartesian, and morphisms `φ' : a ⟶ b'`, `ψ : b ⟶ b'` such that
`g ≫ ψ = φ'`. Then `ψ` is the map induced by the universal property. -/
lemma map_uniq (ψ : b ⟶ b') [IsHomLift p g ψ] (hψ : φ ≫ ψ = φ') : ψ = map p f φ hf' φ' :=
  (Classical.choose_spec <| universal_property p f φ _ _ hf' φ').2 ψ ⟨inferInstance, hψ⟩


/-- Given a diagram
```
a --φ--> b        b'
|        |        |
v        v        v
R --f--> S --g--> S'
```
such that `φ` is strongly cocartesian, and morphisms `ψ ψ' : b ⟶ b'` such that
`g ≫ ψ = φ' = g ≫ ψ'`. Then we have that `ψ = ψ'`. -/
protected lemma ext (φ : a ⟶ b) [IsStronglyCocartesian p f φ] {S' : 𝒮} {b' : 𝒳} (g : S ⟶ S')
    {ψ ψ' : b ⟶ b'} [IsHomLift p g ψ] [IsHomLift p g ψ'] (h : φ ≫ ψ = φ ≫ ψ') : ψ = ψ' := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝² : p.IsStronglyCocartesian f φ
    S' : 𝒮
    b' : 𝒳
    g : Quiver.Hom S S'
    ψ ψ' : Quiver.Hom b b'
    inst✝¹ : p.IsHomLift g ψ
    inst✝ : p.IsHomLift g ψ'
    h : Eq (CategoryTheory.CategoryStruct.comp φ ψ) (CategoryTheory.CategoryStruct …
    ⊢ Eq ψ ψ'
  -/
  rw [map_uniq p f φ (g := g) rfl (φ ≫ ψ) ψ rfl, map_uniq p f φ (g := g) rfl (φ ≫ ψ) ψ' h.symm]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_self : map p f φ (comp_id f).symm φ = 𝟙 b := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝ : p.IsStronglyCocartesian f φ
    ⊢ Eq (CategoryTheory.Functor.IsStronglyCocartesian.map p f φ ⋯ φ) (CategoryThe …
  -/
  subst_hom_lift p f φ; symm
  /-
    case map
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a✝ b✝ : 𝒳
    φ : Quiver.Hom a✝ b✝
    R S : 𝒮
    a b : 𝒳
    inst✝ : p.IsStronglyCocartesian (p.map φ) φ
    ⊢ Eq (CategoryTheory.CategoryStruct.id b✝) (CategoryTheory.Functor.IsStronglyC …
  -/
  apply map_uniq
  /-
    case map.hψ
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a✝ b✝ : 𝒳
    φ : Quiver.Hom a✝ b✝
    R S : 𝒮
    a b : 𝒳
    inst✝ : p.IsStronglyCocartesian (p.map φ) φ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.id b …
  -/
  simp only [comp_id]
  /-
    🎉 no goals
  -/


/-- When its possible to compare the two, the composition of two `IsStronglyCocartesian.map` will
also be given by a `IsStronglyCocartesian.map`. In other words, given diagrams
```
a --φ--> b        b'         b''
|        |        |          |
v        v        v          v
R --f--> S --g--> S' --g'--> S'
```
and
```
a --φ'--> b'
|         |
v         v
R --f'--> S'

```
and
```
a --φ''--> b''
|          |
v          v
R --f''--> S''
```
such that `φ` and `φ'` are strongly cocartesian morphisms, and such that `f' = f ≫ g` and
`f'' = f' ≫ g'`. Then composing the induced map from `b ⟶ b'` with the induced map from
`b' ⟶ b''` gives the induced map from `b ⟶ b''`. -/
@[reassoc (attr := simp)]
lemma map_comp_map {S' S'' : 𝒮} {b' b'' : 𝒳} {f' : R ⟶ S'} {f'' : R ⟶ S''} {g : S ⟶ S'}
    {g' : S' ⟶ S''} (H : f' = f ≫ g) (H' : f'' = f' ≫ g') (φ' : a ⟶ b') (φ'' : a ⟶ b'')
    [IsStronglyCocartesian p f' φ'] [IsHomLift p f'' φ''] :
    map p f φ H φ' ≫ map p f' φ' H' φ'' =
                                            /-
                                              𝒮 : Type u₁
                                              𝒳 : Type u₂
                                              inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
                                              inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
                                              p : CategoryTheory.Functor 𝒳 𝒮
                                              R S : 𝒮
                                              a b : 𝒳
                                              f : Quiver.Hom R S
                                              φ : Quiver.Hom a b
                                              inst✝² : p.IsStronglyCocartesian f φ
                                              S' S'' : 𝒮
                                              b' b'' : 𝒳
                                              f' : Quiver.Hom R S'
                                              f'' : Quiver.Hom R S''
                                              g : Quiver.Hom S S'
                                              g' : Quiver.Hom S' S''
                                              H : Eq f' (CategoryTheory.CategoryStruct.comp f g)
                                              H' : Eq f'' (CategoryTheory.CategoryStruct.comp f' g')
                                              φ' : Quiver.Hom a b'
                                              φ'' : Quiver.Hom a b''
                                              inst✝¹ : p.IsStronglyCocartesian f' φ'
                                              inst✝ : p.IsHomLift f'' φ''
                                              ⊢ Eq f'' (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct. …
                                            -/
      map p f φ (show f'' = f ≫ (g ≫ g') by rwa [← assoc, ← H]) φ'' := by
                                            /-
                                              🎉 no goals
                                            -/
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝² : p.IsStronglyCocartesian f φ
    S' S'' : 𝒮
    b' b'' : 𝒳
    f' : Quiver.Hom R S'
    f'' : Quiver.Hom R S''
    g : Quiver.Hom S S'
    g' : Quiver.Hom S' S''
    H : Eq f' (CategoryTheory.CategoryStruct.comp f g)
    H' : Eq f'' (CategoryTheory.CategoryStruct.comp f' g')
    φ' : Quiver.Hom a b'
    φ'' : Quiver.Hom a b''
    inst✝¹ : p.IsStronglyCocartesian f' φ'
    inst✝ : p.IsHomLift f'' φ''
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsStronglyCoc …
  -/
  apply map_uniq p f φ
  /-
    case hψ
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝² : p.IsStronglyCocartesian f φ
    S' S'' : 𝒮
    b' b'' : 𝒳
    f' : Quiver.Hom R S'
    f'' : Quiver.Hom R S''
    g : Quiver.Hom S S'
    g' : Quiver.Hom S' S''
    H : Eq f' (CategoryTheory.CategoryStruct.comp f g)
    H' : Eq f'' (CategoryTheory.CategoryStruct.comp f' g')
    φ' : Quiver.Hom a b'
    φ'' : Quiver.Hom a b''
    inst✝¹ : p.IsStronglyCocartesian f' φ'
    inst✝ : p.IsHomLift f'' φ''
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
  -/
  simp only [fac_assoc, fac]
  /-
    🎉 no goals
  -/


/-- Given two strongly cocartesian morphisms `φ`, `ψ` as follows
```
a --φ--> b --ψ--> c
|        |        |
v        v        v
R --f--> S --g--> T
```
Then the composite `φ ≫ ψ` is also strongly cocartesian. -/
instance comp [IsStronglyCocartesian p f φ] [IsStronglyCocartesian p g ψ] :
    IsStronglyCocartesian p (f ≫ g) (φ ≫ ψ) where
  universal_property' := by
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S T : 𝒮
      a b c : 𝒳
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      φ : Quiver.Hom a b
      ψ : Quiver.Hom b c
      inst✝¹ : p.IsStronglyCocartesian f φ
      inst✝ : p.IsStronglyCocartesian g ψ
      ⊢ ∀ {b' : 𝒳} (g_1 : Quiver.Hom T (p.obj b')) (φ' : Quiver.Hom a b') [inst : p. …
    -/
    intro c' h τ hτ
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S T : 𝒮
      a b c : 𝒳
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      φ : Quiver.Hom a b
      ψ : Quiver.Hom b c
      inst✝¹ : p.IsStronglyCocartesian f φ
      inst✝ : p.IsStronglyCocartesian g ψ
      c' : 𝒳
      h : Quiver.Hom T (p.obj c')
      τ : Quiver.Hom a c'
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryS …
      ⊢ ExistsUnique fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory.CategoryStru …
    -/
    use map p g ψ (f' := g ≫ h) rfl <| map p f φ (assoc f g h) τ
    /-
      case h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S T : 𝒮
      a b c : 𝒳
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      φ : Quiver.Hom a b
      ψ : Quiver.Hom b c
      inst✝¹ : p.IsStronglyCocartesian f φ
      inst✝ : p.IsStronglyCocartesian g ψ
      c' : 𝒳
      h : Quiver.Hom T (p.obj c')
      τ : Quiver.Hom a c'
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryS …
      ⊢ And ((fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory.CategoryStruct.comp …
    -/
    refine ⟨⟨inferInstance, ?_⟩, ?_⟩
      /-
        case h.refine_1
        𝒮 : Type u₁
        𝒳 : Type u₂
        inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
        inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
        p : CategoryTheory.Functor 𝒳 𝒮
        R S T : 𝒮
        a b c : 𝒳
        f : Quiver.Hom R S
        g : Quiver.Hom S T
        φ : Quiver.Hom a b
        ψ : Quiver.Hom b c
        inst✝¹ : p.IsStronglyCocartesian f φ
        inst✝ : p.IsStronglyCocartesian g ψ
        c' : 𝒳
        h : Quiver.Hom T (p.obj c')
        τ : Quiver.Hom a c'
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryS …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp φ …
      -/
    · simp only [assoc, fac]
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2
        𝒮 : Type u₁
        𝒳 : Type u₂
        inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
        inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
        p : CategoryTheory.Functor 𝒳 𝒮
        R S T : 𝒮
        a b c : 𝒳
        f : Quiver.Hom R S
        g : Quiver.Hom S T
        φ : Quiver.Hom a b
        ψ : Quiver.Hom b c
        inst✝¹ : p.IsStronglyCocartesian f φ
        inst✝ : p.IsStronglyCocartesian g ψ
        c' : 𝒳
        h : Quiver.Hom T (p.obj c')
        τ : Quiver.Hom a c'
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryS …
        ⊢ ∀ (y : Quiver.Hom c c'), (fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory …
      -/
    · intro π' ⟨hπ'₁, hπ'₂⟩
      /-
        case h.refine_2
        𝒮 : Type u₁
        𝒳 : Type u₂
        inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
        inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
        p : CategoryTheory.Functor 𝒳 𝒮
        R S T : 𝒮
        a b c : 𝒳
        f : Quiver.Hom R S
        g : Quiver.Hom S T
        φ : Quiver.Hom a b
        ψ : Quiver.Hom b c
        inst✝¹ : p.IsStronglyCocartesian f φ
        inst✝ : p.IsStronglyCocartesian g ψ
        c' : 𝒳
        h : Quiver.Hom T (p.obj c')
        τ : Quiver.Hom a c'
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryS …
        π' : Quiver.Hom c c'
        hπ'₁ : p.IsHomLift h π'
        hπ'₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
        ⊢ Eq π' (CategoryTheory.Functor.IsStronglyCocartesian.map p g ψ ⋯ (CategoryThe …
      -/
      apply map_uniq
      /-
        case h.refine_2.hψ
        𝒮 : Type u₁
        𝒳 : Type u₂
        inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
        inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
        p : CategoryTheory.Functor 𝒳 𝒮
        R S T : 𝒮
        a b c : 𝒳
        f : Quiver.Hom R S
        g : Quiver.Hom S T
        φ : Quiver.Hom a b
        ψ : Quiver.Hom b c
        inst✝¹ : p.IsStronglyCocartesian f φ
        inst✝ : p.IsStronglyCocartesian g ψ
        c' : 𝒳
        h : Quiver.Hom T (p.obj c')
        τ : Quiver.Hom a c'
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryS …
        π' : Quiver.Hom c c'
        hπ'₁ : p.IsHomLift h π'
        hπ'₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ψ π') (CategoryTheory.Functor.IsStron …
      -/
      apply map_uniq
      /-
        case h.refine_2.hψ.hψ
        𝒮 : Type u₁
        𝒳 : Type u₂
        inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
        inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
        p : CategoryTheory.Functor 𝒳 𝒮
        R S T : 𝒮
        a b c : 𝒳
        f : Quiver.Hom R S
        g : Quiver.Hom S T
        φ : Quiver.Hom a b
        ψ : Quiver.Hom b c
        inst✝¹ : p.IsStronglyCocartesian f φ
        inst✝ : p.IsStronglyCocartesian g ψ
        c' : 𝒳
        h : Quiver.Hom T (p.obj c')
        τ : Quiver.Hom a c'
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryS …
        π' : Quiver.Hom c c'
        hπ'₁ : p.IsHomLift h π'
        hπ'₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
      -/
      simp only [← hπ'₂, assoc]
      /-
        🎉 no goals
      -/


/-- Given two commutative squares
```
a --φ--> b --ψ--> c
|        |        |
v        v        v
R --f--> S --g--> T
```
such that `φ ≫ ψ` and `φ` are strongly cocartesian, then so is `ψ`. -/
protected lemma of_comp [IsStronglyCocartesian p f φ] [IsStronglyCocartesian p (f ≫ g) (φ ≫ ψ)]
    [IsHomLift p g ψ] : IsStronglyCocartesian p g ψ where
  universal_property' := by
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S T : 𝒮
      a b c : 𝒳
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      φ : Quiver.Hom a b
      ψ : Quiver.Hom b c
      inst✝² : p.IsStronglyCocartesian f φ
      inst✝¹ : p.IsStronglyCocartesian (CategoryTheory.CategoryStruct.comp f g) (Cat …
      inst✝ : p.IsHomLift g ψ
      ⊢ ∀ {b' : 𝒳} (g_1 : Quiver.Hom T (p.obj b')) (φ' : Quiver.Hom b b') [inst : p. …
    -/
    intro c' h τ hτ
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S T : 𝒮
      a b c : 𝒳
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      φ : Quiver.Hom a b
      ψ : Quiver.Hom b c
      inst✝² : p.IsStronglyCocartesian f φ
      inst✝¹ : p.IsStronglyCocartesian (CategoryTheory.CategoryStruct.comp f g) (Cat …
      inst✝ : p.IsHomLift g ψ
      c' : 𝒳
      h : Quiver.Hom T (p.obj c')
      τ : Quiver.Hom b c'
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp g h) τ
      ⊢ ExistsUnique fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory.CategoryStru …
    -/
    have h₁ : IsHomLift p (f ≫ g ≫ h) (φ ≫ τ) := by simpa using IsHomLift.comp p f (g ≫ h) φ τ
    /- We get a morphism `π : c ⟶ c'` such that `(φ ≫ ψ) ≫ π = φ ≫ τ` from the universal property
    of `φ ≫ ψ`. This will be the morphism induced by `φ`. -/
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S T : 𝒮
      a b c : 𝒳
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      φ : Quiver.Hom a b
      ψ : Quiver.Hom b c
      inst✝² : p.IsStronglyCocartesian f φ
      inst✝¹ : p.IsStronglyCocartesian (CategoryTheory.CategoryStruct.comp f g) (Cat …
      inst✝ : p.IsHomLift g ψ
      c' : 𝒳
      h : Quiver.Hom T (p.obj c')
      τ : Quiver.Hom b c'
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp g h) τ
      h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Categor …
      ⊢ ExistsUnique fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory.CategoryStru …
    -/
    use map p (f ≫ g) (φ ≫ ψ) (f' := f ≫ g ≫ h) (assoc f g h).symm (φ ≫ τ)
    /-
      case h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S T : 𝒮
      a b c : 𝒳
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      φ : Quiver.Hom a b
      ψ : Quiver.Hom b c
      inst✝² : p.IsStronglyCocartesian f φ
      inst✝¹ : p.IsStronglyCocartesian (CategoryTheory.CategoryStruct.comp f g) (Cat …
      inst✝ : p.IsHomLift g ψ
      c' : 𝒳
      h : Quiver.Hom T (p.obj c')
      τ : Quiver.Hom b c'
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp g h) τ
      h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Categor …
      ⊢ And ((fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory.CategoryStruct.comp …
    -/
    refine ⟨⟨inferInstance, ?_⟩, ?_⟩
    /- The fact that `ψ ≫ π = τ` follows from `φ ≫ ψ ≫ π = φ ≫ τ` and the universal property of
    `φ`. -/
      /-
        case h.refine_1
        𝒮 : Type u₁
        𝒳 : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
        inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
        p : CategoryTheory.Functor 𝒳 𝒮
        R S T : 𝒮
        a b c : 𝒳
        f : Quiver.Hom R S
        g : Quiver.Hom S T
        φ : Quiver.Hom a b
        ψ : Quiver.Hom b c
        inst✝² : p.IsStronglyCocartesian f φ
        inst✝¹ : p.IsStronglyCocartesian (CategoryTheory.CategoryStruct.comp f g) (Cat …
        inst✝ : p.IsHomLift g ψ
        c' : 𝒳
        h : Quiver.Hom T (p.obj c')
        τ : Quiver.Hom b c'
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp g h) τ
        h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Categor …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ψ (CategoryTheory.Functor.IsStronglyC …
      -/
    · apply IsStronglyCocartesian.ext p f φ (g ≫ h) <| by simp only [← assoc, fac]
      /-
        🎉 no goals
      -/
    -- Finally, uniqueness of `π` comes from the universal property of `φ ≫ ψ`.
      /-
        case h.refine_2
        𝒮 : Type u₁
        𝒳 : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
        inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
        p : CategoryTheory.Functor 𝒳 𝒮
        R S T : 𝒮
        a b c : 𝒳
        f : Quiver.Hom R S
        g : Quiver.Hom S T
        φ : Quiver.Hom a b
        ψ : Quiver.Hom b c
        inst✝² : p.IsStronglyCocartesian f φ
        inst✝¹ : p.IsStronglyCocartesian (CategoryTheory.CategoryStruct.comp f g) (Cat …
        inst✝ : p.IsHomLift g ψ
        c' : 𝒳
        h : Quiver.Hom T (p.obj c')
        τ : Quiver.Hom b c'
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp g h) τ
        h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Categor …
        ⊢ ∀ (y : Quiver.Hom c c'), (fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory …
      -/
    · intro π' ⟨hπ'₁, hπ'₂⟩
      /-
        case h.refine_2
        𝒮 : Type u₁
        𝒳 : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
        inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
        p : CategoryTheory.Functor 𝒳 𝒮
        R S T : 𝒮
        a b c : 𝒳
        f : Quiver.Hom R S
        g : Quiver.Hom S T
        φ : Quiver.Hom a b
        ψ : Quiver.Hom b c
        inst✝² : p.IsStronglyCocartesian f φ
        inst✝¹ : p.IsStronglyCocartesian (CategoryTheory.CategoryStruct.comp f g) (Cat …
        inst✝ : p.IsHomLift g ψ
        c' : 𝒳
        h : Quiver.Hom T (p.obj c')
        τ : Quiver.Hom b c'
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp g h) τ
        h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Categor …
        π' : Quiver.Hom c c'
        hπ'₁ : p.IsHomLift h π'
        hπ'₂ : Eq (CategoryTheory.CategoryStruct.comp ψ π') τ
        ⊢ Eq π' (CategoryTheory.Functor.IsStronglyCocartesian.map p (CategoryTheory.Ca …
      -/
      apply map_uniq
      /-
        case h.refine_2.hψ
        𝒮 : Type u₁
        𝒳 : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} 𝒮
        inst✝³ : CategoryTheory.Category.{v₂, u₂} 𝒳
        p : CategoryTheory.Functor 𝒳 𝒮
        R S T : 𝒮
        a b c : 𝒳
        f : Quiver.Hom R S
        g : Quiver.Hom S T
        φ : Quiver.Hom a b
        ψ : Quiver.Hom b c
        inst✝² : p.IsStronglyCocartesian f φ
        inst✝¹ : p.IsStronglyCocartesian (CategoryTheory.CategoryStruct.comp f g) (Cat …
        inst✝ : p.IsHomLift g ψ
        c' : 𝒳
        h : Quiver.Hom T (p.obj c')
        τ : Quiver.Hom b c'
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp g h) τ
        h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Categor …
        π' : Quiver.Hom c c'
        hπ'₁ : p.IsHomLift h π'
        hπ'₂ : Eq (CategoryTheory.CategoryStruct.comp ψ π') τ
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp φ …
      -/
      simp [hπ'₂.symm]
      /-
        🎉 no goals
      -/


instance of_iso (φ : a ≅ b) [IsHomLift p f φ.hom] : IsStronglyCocartesian p f φ.hom where
  universal_property' := by
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : CategoryTheory.Iso a b
      inst✝ : p.IsHomLift f φ.hom
      ⊢ ∀ {b' : 𝒳} (g : Quiver.Hom S (p.obj b')) (φ' : Quiver.Hom a b') [inst : p.Is …
    -/
    intro b' g τ hτ
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : CategoryTheory.Iso a b
      inst✝ : p.IsHomLift f φ.hom
      b' : 𝒳
      g : Quiver.Hom S (p.obj b')
      τ : Quiver.Hom a b'
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp f g) τ
      ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
    -/
    use φ.inv ≫ τ
    /-
      case h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : CategoryTheory.Iso a b
      inst✝ : p.IsHomLift f φ.hom
      b' : 𝒳
      g : Quiver.Hom S (p.obj b')
      τ : Quiver.Hom a b'
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp f g) τ
      ⊢ And ((fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStruct.comp …
    -/
    refine ⟨?_, by aesop_cat⟩
    /-
      case h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      R S : 𝒮
      a b : 𝒳
      f : Quiver.Hom R S
      φ : CategoryTheory.Iso a b
      inst✝ : p.IsHomLift f φ.hom
      b' : 𝒳
      g : Quiver.Hom S (p.obj b')
      τ : Quiver.Hom a b'
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp f g) τ
      ⊢ (fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStruct.comp φ.ho …
    -/
    simpa [← assoc] using (IsHomLift.comp p (isoOfIsoLift p f φ).inv (f ≫ g) φ.inv τ)
    /-
      🎉 no goals
    -/


instance of_isIso (φ : a ⟶ b) [IsHomLift p f φ] [IsIso φ] : IsStronglyCocartesian p f φ :=
                                                                  /-
                                                                    𝒮 : Type u₁
                                                                    𝒳 : Type u₂
                                                                    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
                                                                    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
                                                                    p : CategoryTheory.Functor 𝒳 𝒮
                                                                    R S : 𝒮
                                                                    a b : 𝒳
                                                                    f : Quiver.Hom R S
                                                                    φ : Quiver.Hom a b
                                                                    inst✝¹ : p.IsHomLift f φ
                                                                    inst✝ : CategoryTheory.IsIso φ
                                                                    ⊢ p.IsHomLift f (CategoryTheory.asIso φ).hom
                                                                  -/
  @IsStronglyCocartesian.of_iso _ _ _ _ p _ _ _ _ f (asIso φ) (by aesop)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- A strongly cocartesian arrow lying over an isomorphism is an isomorphism. -/
lemma isIso_of_base_isIso (φ : a ⟶ b) [IsStronglyCocartesian p f φ] [IsIso f] : IsIso φ := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCocartesian f φ
    inst✝ : CategoryTheory.IsIso f
    ⊢ CategoryTheory.IsIso φ
  -/
  subst_hom_lift p f φ; clear a b R S
  -- Let `φ'` be the morphism induced by applying universal property to `𝟙 a` lying over `f ≫ f⁻¹`.
  /-
    case map
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCocartesian (p.map φ) φ
    inst✝ : CategoryTheory.IsIso (p.map φ)
    ⊢ CategoryTheory.IsIso φ
  -/
  let φ' := map p (p.map φ) φ (IsIso.hom_inv_id (p.map φ)).symm (𝟙 a)
  /-
    case map
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCocartesian (p.map φ) φ
    inst✝ : CategoryTheory.IsIso (p.map φ)
    φ' : Quiver.Hom b a := CategoryTheory.Functor.IsStronglyCocartesian.map p (p.m …
    ⊢ CategoryTheory.IsIso φ
  -/
  use φ'
  -- `φ ≫ φ' = 𝟙 a` follows immediately from the universal property.
  /-
    case h
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCocartesian (p.map φ) φ
    inst✝ : CategoryTheory.IsIso (p.map φ)
    φ' : Quiver.Hom b a := CategoryTheory.Functor.IsStronglyCocartesian.map p (p.m …
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp φ φ') (CategoryTheory.CategorySt …
  -/
  have inv_hom : φ ≫ φ' = 𝟙 a := fac p (p.map φ) φ _ (𝟙 a)
  /-
    case h
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCocartesian (p.map φ) φ
    inst✝ : CategoryTheory.IsIso (p.map φ)
    φ' : Quiver.Hom b a := CategoryTheory.Functor.IsStronglyCocartesian.map p (p.m …
    inv_hom : Eq (CategoryTheory.CategoryStruct.comp φ φ') (CategoryTheory.Categor …
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp φ φ') (CategoryTheory.CategorySt …
  -/
  refine ⟨inv_hom, ?_⟩
  -- We will now show that `φ' ≫ φ = 𝟙 b` by showing that `φ ≫ (φ' ≫ φ) = φ ≫ 𝟙 b`.
  have h₁ : IsHomLift p (𝟙 (p.obj b)) (φ' ≫ φ) := by
    rw [← IsIso.inv_hom_id (p.map φ)]
    apply IsHomLift.comp
  /-
    case h
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCocartesian (p.map φ) φ
    inst✝ : CategoryTheory.IsIso (p.map φ)
    φ' : Quiver.Hom b a := CategoryTheory.Functor.IsStronglyCocartesian.map p (p.m …
    inv_hom : Eq (CategoryTheory.CategoryStruct.comp φ φ') (CategoryTheory.Categor …
    h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id (p.obj b)) (CategoryTheory. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ' φ) (CategoryTheory.CategoryStruct. …
  -/
  apply IsStronglyCocartesian.ext p (p.map φ) φ (𝟙 (p.obj b))
  /-
    case h
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCocartesian (p.map φ) φ
    inst✝ : CategoryTheory.IsIso (p.map φ)
    φ' : Quiver.Hom b a := CategoryTheory.Functor.IsStronglyCocartesian.map p (p.m …
    inv_hom : Eq (CategoryTheory.CategoryStruct.comp φ φ') (CategoryTheory.Categor …
    h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id (p.obj b)) (CategoryTheory. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
  -/
  simp only [← assoc, inv_hom, comp_id, id_comp]
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism between the codomains of two strongly cocartesian arrows lying over
isomorphic objects. -/
noncomputable def codomainIsoOfBaseIso {R S S' : 𝒮} {a b b' : 𝒳} {f : R ⟶ S} {f' : R ⟶ S'}
  {g : S ≅ S'} (h : f' = f ≫ g.hom) (φ : a ⟶ b) (φ' : a ⟶ b') [IsStronglyCocartesian p f φ]
    [IsStronglyCocartesian p f' φ'] : b ≅ b' where
  hom := map p f φ h φ'
  inv := @map _ _ _ _ p _ _ _ _ f' φ' _ _ _ _ _ (congrArg (· ≫ g.inv) h.symm) φ
        /-
          𝒮 : Type u₁
          𝒳 : Type u₂
          inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
          inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
          p : CategoryTheory.Functor 𝒳 𝒮
          R S S' : 𝒮
          a b b' : 𝒳
          f : Quiver.Hom R S
          f' : Quiver.Hom R S'
          g : CategoryTheory.Iso S S'
          h : Eq f' (CategoryTheory.CategoryStruct.comp f g.hom)
          φ : Quiver.Hom a b
          φ' : Quiver.Hom a b'
          inst✝¹ : p.IsStronglyCocartesian f φ
          inst✝ : p.IsStronglyCocartesian f' φ'
          ⊢ p.IsHomLift ((fun x => CategoryTheory.CategoryStruct.comp x g.inv) (Category …
        -/
    (by simp; infer_instance)
              /-
                🎉 no goals
              -/


