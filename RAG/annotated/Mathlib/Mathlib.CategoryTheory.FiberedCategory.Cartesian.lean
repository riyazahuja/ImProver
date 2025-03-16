/-- A morphism `φ : a ⟶ b` in `𝒳` lying over `f : R ⟶ S` in `𝒮` is cartesian if for all
morphisms `φ' : a' ⟶ b`, also lying over `f`, there exists a unique morphism `χ : a' ⟶ a` lifting
`𝟙 R` such that `φ' = χ ≫ φ`.

See SGA 1 VI 5.1. -/
class IsCartesian extends IsHomLift p f φ : Prop where
  universal_property {a' : 𝒳} (φ' : a' ⟶ b) [IsHomLift p f φ'] :
      ∃! χ : a' ⟶ a, IsHomLift p (𝟙 R) χ ∧ χ ≫ φ = φ'


/-- A morphism `φ : a ⟶ b` in `𝒳` lying over `f : R ⟶ S` in `𝒮` is strongly cartesian if for
all morphisms `φ' : a' ⟶ b` and all diagrams of the form
```
a'        a --φ--> b
|         |        |
v         v        v
R' --g--> R --f--> S
```
such that `φ'` lifts `g ≫ f`, there exists a lift `χ` of `g` such that `φ' = χ ≫ φ`.

See <https://stacks.math.columbia.edu/tag/02XK>. -/
class IsStronglyCartesian extends IsHomLift p f φ : Prop where
  universal_property' {a' : 𝒳} (g : p.obj a' ⟶ R) (φ' : a' ⟶ b) [IsHomLift p (g ≫ f) φ'] :
      ∃! χ : a' ⟶ a, IsHomLift p g χ ∧ χ ≫ φ = φ'


/-- Given a cartesian morphism `φ : a ⟶ b` lying over `f : R ⟶ S` in `𝒳`, and another morphism
`φ' : a' ⟶ b` which also lifts `f`, then `IsCartesian.map f φ φ'` is the morphism `a' ⟶ a` lifting
`𝟙 R` obtained from the universal property of `φ`. -/
protected noncomputable def map : a' ⟶ a :=
  Classical.choose <| IsCartesian.universal_property (p := p) (f := f) (φ := φ) φ'


instance map_isHomLift : IsHomLift p (𝟙 R) (IsCartesian.map p f φ φ') :=
  (Classical.choose_spec <| IsCartesian.universal_property (p := p) (f := f) (φ := φ) φ').1.1


@[reassoc (attr := simp)]
lemma fac : IsCartesian.map p f φ φ' ≫ φ = φ' :=
  (Classical.choose_spec <| IsCartesian.universal_property (p := p) (f := f) (φ := φ) φ').1.2


/-- Given a cartesian morphism `φ : a ⟶ b` lying over `f : R ⟶ S` in `𝒳`, and another morphism
`φ' : a' ⟶ b` which also lifts `f`. Then any morphism `ψ : a' ⟶ a` lifting `𝟙 R` such that
`g ≫ ψ = φ'` must equal the map induced from the universal property of `φ`. -/
lemma map_uniq (ψ : a' ⟶ a) [IsHomLift p (𝟙 R) ψ] (hψ : ψ ≫ φ = φ') :
    ψ = IsCartesian.map p f φ φ' :=
  (Classical.choose_spec <| IsCartesian.universal_property (p := p) (f := f) (φ := φ) φ').2
    ψ ⟨inferInstance, hψ⟩


/-- Given a cartesian morphism `φ : a ⟶ b` lying over `f : R ⟶ S` in `𝒳`, and two morphisms
`ψ ψ' : a' ⟶ a` such that `ψ ≫ φ = ψ' ≫ φ`. Then we must have `ψ = ψ'`. -/
protected lemma ext (φ : a ⟶ b) [IsCartesian p f φ] {a' : 𝒳} (ψ ψ' : a' ⟶ a)
    [IsHomLift p (𝟙 R) ψ] [IsHomLift p (𝟙 R) ψ'] (h : ψ ≫ φ = ψ' ≫ φ) : ψ = ψ' := by
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
    inst✝² : p.IsCartesian f φ
    a' : 𝒳
    ψ ψ' : Quiver.Hom a' a
    inst✝¹ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) ψ
    inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) ψ'
    h : Eq (CategoryTheory.CategoryStruct.comp ψ φ) (CategoryTheory.CategoryStruct …
    ⊢ Eq ψ ψ'
  -/
  rw [map_uniq p f φ (ψ ≫ φ) ψ rfl, map_uniq p f φ (ψ ≫ φ) ψ' h.symm]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_self : IsCartesian.map p f φ φ = 𝟙 a := by
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
    inst✝ : p.IsCartesian f φ
    ⊢ Eq (CategoryTheory.Functor.IsCartesian.map p f φ φ) (CategoryTheory.Category …
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
    inst✝ : p.IsCartesian (p.map φ) φ
    ⊢ Eq (CategoryTheory.CategoryStruct.id a✝) (CategoryTheory.Functor.IsCartesian …
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
    inst✝ : p.IsCartesian (p.map φ) φ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id a✝) …
  -/
  simp only [id_comp]
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism between the domains of two cartesian morphisms
lying over the same object. -/
@[simps]
noncomputable def domainUniqueUpToIso {a' : 𝒳} (φ' : a' ⟶ b) [IsCartesian p f φ'] : a' ≅ a where
  hom := IsCartesian.map p f φ φ'
  inv := IsCartesian.map p f φ' φ
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
      inst✝¹ : p.IsCartesian f φ
      a' : 𝒳
      φ' : Quiver.Hom a' b
      inst✝ : p.IsCartesian f φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCartesian.m …
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
      a' : 𝒳
      φ' : Quiver.Hom a' b✝
      R S : 𝒮
      a b : 𝒳
      inst✝¹ : p.IsCartesian (p.map φ') φ
      inst✝ : p.IsCartesian (p.map φ') φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCartesian.m …
    -/
    apply IsCartesian.ext p (p.map φ') φ'
    /-
      case map.h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      a✝ b✝ : 𝒳
      φ : Quiver.Hom a✝ b✝
      a' : 𝒳
      φ' : Quiver.Hom a' b✝
      R S : 𝒮
      a b : 𝒳
      inst✝¹ : p.IsCartesian (p.map φ') φ
      inst✝ : p.IsCartesian (p.map φ') φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [assoc, fac, id_comp]
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
      inst✝¹ : p.IsCartesian f φ
      a' : 𝒳
      φ' : Quiver.Hom a' b
      inst✝ : p.IsCartesian f φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCartesian.m …
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
      a' : 𝒳
      φ' : Quiver.Hom a' b✝
      R S : 𝒮
      a b : 𝒳
      inst✝¹ : p.IsCartesian (p.map φ) φ
      inst✝ : p.IsCartesian (p.map φ) φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCartesian.m …
    -/
    apply IsCartesian.ext p (p.map φ) φ
    /-
      case map.h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      a✝ b✝ : 𝒳
      φ : Quiver.Hom a✝ b✝
      a' : 𝒳
      φ' : Quiver.Hom a' b✝
      R S : 𝒮
      a b : 𝒳
      inst✝¹ : p.IsCartesian (p.map φ) φ
      inst✝ : p.IsCartesian (p.map φ) φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [assoc, fac, id_comp]
    /-
      🎉 no goals
    -/


instance domainUniqueUpToIso_inv_isHomLift {a' : 𝒳} (φ' : a' ⟶ b) [IsCartesian p f φ'] :
    IsHomLift p (𝟙 R) (domainUniqueUpToIso p f φ φ').hom :=
  domainUniqueUpToIso_hom p f φ φ' ▸ IsCartesian.map_isHomLift p f φ φ'


instance domainUniqueUpToIso_hom_isHomLift {a' : 𝒳} (φ' : a' ⟶ b) [IsCartesian p f φ'] :
    IsHomLift p (𝟙 R) (domainUniqueUpToIso p f φ φ').inv :=
  domainUniqueUpToIso_inv p f φ φ' ▸ IsCartesian.map_isHomLift p f φ' φ


/-- Precomposing a cartesian morphism with an isomorphism lifting the identity is cartesian. -/
instance of_iso_comp {a' : 𝒳} (φ' : a' ≅ a) [IsHomLift p (𝟙 R) φ'.hom] :
    IsCartesian p f (φ'.hom ≫ φ) where
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
      inst✝¹ : p.IsCartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      ⊢ ∀ {a'_1 : 𝒳} (φ'_1 : Quiver.Hom a'_1 b) [inst : p.IsHomLift f φ'_1], ExistsU …
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
      inst✝¹ : p.IsCartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom c b
      hψ : p.IsHomLift f ψ
      ⊢ ExistsUnique fun χ => And (p.IsHomLift (CategoryTheory.CategoryStruct.id R)  …
    -/
    use IsCartesian.map p f φ ψ ≫ φ'.inv
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
      inst✝¹ : p.IsCartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom c b
      hψ : p.IsHomLift f ψ
      ⊢ And ((fun χ => And (p.IsHomLift (CategoryTheory.CategoryStruct.id R) χ) (Eq  …
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
      inst✝¹ : p.IsCartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom c b
      hψ : p.IsHomLift f ψ
      ⊢ ∀ (y : Quiver.Hom c a'), (fun χ => And (p.IsHomLift (CategoryTheory.Category …
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
      inst✝¹ : p.IsCartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom c b
      hψ : p.IsHomLift f ψ
      τ : Quiver.Hom c a'
      hτ₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) τ
      hτ₂ : Eq (CategoryTheory.CategoryStruct.comp τ (CategoryTheory.CategoryStruct. …
      ⊢ Eq τ (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCartesian …
    -/
    rw [Iso.eq_comp_inv]
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
      inst✝¹ : p.IsCartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom c b
      hψ : p.IsHomLift f ψ
      τ : Quiver.Hom c a'
      hτ₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) τ
      hτ₂ : Eq (CategoryTheory.CategoryStruct.comp τ (CategoryTheory.CategoryStruct. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp τ φ'.hom) (CategoryTheory.Functor.IsC …
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
      inst✝¹ : p.IsCartesian f φ
      a' : 𝒳
      φ' : CategoryTheory.Iso a' a
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom c b
      hψ : p.IsHomLift f ψ
      τ : Quiver.Hom c a'
      hτ₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) τ
      hτ₂ : Eq (CategoryTheory.CategoryStruct.comp τ (CategoryTheory.CategoryStruct. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp τ …
    -/
    simp only [assoc, hτ₂]
    /-
      🎉 no goals
    -/


/-- Postcomposing a cartesian morphism with an isomorphism lifting the identity is cartesian. -/
instance of_comp_iso {b' : 𝒳} (φ' : b ≅ b') [IsHomLift p (𝟙 S) φ'.hom] :
    IsCartesian p f (φ ≫ φ'.hom) where
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
      inst✝¹ : p.IsCartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      ⊢ ∀ {a' : 𝒳} (φ'_1 : Quiver.Hom a' b') [inst : p.IsHomLift f φ'_1], ExistsUniq …
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
      inst✝¹ : p.IsCartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom c b'
      hψ : p.IsHomLift f ψ
      ⊢ ExistsUnique fun χ => And (p.IsHomLift (CategoryTheory.CategoryStruct.id R)  …
    -/
    use IsCartesian.map p f φ (ψ ≫ φ'.inv)
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
      inst✝¹ : p.IsCartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom c b'
      hψ : p.IsHomLift f ψ
      ⊢ And ((fun χ => And (p.IsHomLift (CategoryTheory.CategoryStruct.id R) χ) (Eq  …
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
      inst✝¹ : p.IsCartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom c b'
      hψ : p.IsHomLift f ψ
      ⊢ ∀ (y : Quiver.Hom c a), (fun χ => And (p.IsHomLift (CategoryTheory.CategoryS …
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
      inst✝¹ : p.IsCartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom c b'
      hψ : p.IsHomLift f ψ
      τ : Quiver.Hom c a
      hτ₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) τ
      hτ₂ : Eq (CategoryTheory.CategoryStruct.comp τ (CategoryTheory.CategoryStruct. …
      ⊢ Eq τ (CategoryTheory.Functor.IsCartesian.map p f φ (CategoryTheory.CategoryS …
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
      inst✝¹ : p.IsCartesian f φ
      b' : 𝒳
      φ' : CategoryTheory.Iso b b'
      inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ'.hom
      c : 𝒳
      ψ : Quiver.Hom c b'
      hψ : p.IsHomLift f ψ
      τ : Quiver.Hom c a
      hτ₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) τ
      hτ₂ : Eq (CategoryTheory.CategoryStruct.comp τ (CategoryTheory.CategoryStruct. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp τ φ) (CategoryTheory.CategoryStruct.c …
    -/
    simp only [Iso.eq_comp_inv, assoc, hτ₂]
    /-
      🎉 no goals
    -/


/-- The universal property of a strongly cartesian morphism.

This lemma is more flexible with respect to non-definitional equalities than the field
`universal_property'` of `IsStronglyCartesian`. -/
lemma universal_property {R' : 𝒮} {a' : 𝒳} (g : R' ⟶ R) (f' : R' ⟶ S) (hf' : f' = g ≫ f)
    (φ' : a' ⟶ b) [IsHomLift p f' φ'] : ∃! χ : a' ⟶ a, IsHomLift p g χ ∧ χ ≫ φ = φ' := by
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
    inst✝¹ : p.IsStronglyCartesian f φ
    R' : 𝒮
    a' : 𝒳
    g : Quiver.Hom R' R
    f' : Quiver.Hom R' S
    hf' : Eq f' (CategoryTheory.CategoryStruct.comp g f)
    φ' : Quiver.Hom a' b
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
    R : 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    a' : 𝒳
    φ' : Quiver.Hom a' b
    g : Quiver.Hom (p.obj a') R
    f : Quiver.Hom R (p.obj b)
    inst✝¹ : p.IsStronglyCartesian f φ
    hf' : Eq (p.map φ') (CategoryTheory.CategoryStruct.comp g f)
    inst✝ : p.IsHomLift (p.map φ') φ'
    ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
  -/
  have : p.IsHomLift (g ≫ f) φ' := (hf' ▸ inferInstance)
  /-
    case map
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    R : 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    a' : 𝒳
    φ' : Quiver.Hom a' b
    g : Quiver.Hom (p.obj a') R
    f : Quiver.Hom R (p.obj b)
    inst✝¹ : p.IsStronglyCartesian f φ
    hf' : Eq (p.map φ') (CategoryTheory.CategoryStruct.comp g f)
    inst✝ : p.IsHomLift (p.map φ') φ'
    this : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) φ'
    ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
  -/
  apply IsStronglyCartesian.universal_property' f
  /-
    🎉 no goals
  -/


instance isCartesian_of_isStronglyCartesian [p.IsStronglyCartesian f φ] : p.IsCartesian f φ where
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
                                                                         inst✝² inst✝¹ : p.IsStronglyCartesian f φ
                                                                         a'✝ : 𝒳
                                                                         φ' : Quiver.Hom a'✝ b
                                                                         inst✝ : p.IsHomLift f φ'
                                                                         ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id R …
                                                                       -/
  universal_property := fun φ' => universal_property p f φ (𝟙 R) f (by simp) φ'
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- Given a diagram
```
a'        a --φ--> b
|         |        |
v         v        v
R' --g--> R --f--> S
```
such that `φ` is strongly cartesian, and a morphism `φ' : a' ⟶ b`. Then `map` is the map `a' ⟶ a`
lying over `g` obtained from the universal property of `φ`. -/
noncomputable def map : a' ⟶ a :=
  Classical.choose <| universal_property p f φ _ _ hf' φ'


instance map_isHomLift : IsHomLift p g (map p f φ hf' φ') :=
  (Classical.choose_spec <| universal_property p f φ _ _ hf' φ').1.1


@[reassoc (attr := simp)]
lemma fac : (map p f φ hf' φ') ≫ φ = φ' :=
  (Classical.choose_spec <| universal_property p f φ _ _ hf' φ').1.2


/-- Given a diagram
```
a'        a --φ--> b
|         |        |
v         v        v
R' --g--> R --f--> S
```
such that `φ` is strongly cartesian, and morphisms `φ' : a' ⟶ b`, `ψ : a' ⟶ a` such that
`ψ ≫ φ = φ'`. Then `ψ` is the map induced by the universal property. -/
lemma map_uniq (ψ : a' ⟶ a) [IsHomLift p g ψ] (hψ : ψ ≫ φ = φ') : ψ = map p f φ hf' φ' :=
  (Classical.choose_spec <| universal_property p f φ _ _ hf' φ').2 ψ ⟨inferInstance, hψ⟩


/-- Given a diagram
```
a'        a --φ--> b
|         |        |
v         v        v
R' --g--> R --f--> S
```
such that `φ` is strongly cartesian, and morphisms `ψ ψ' : a' ⟶ a` such that
`g ≫ ψ = φ' = g ≫ ψ'`. Then we have that `ψ = ψ'`. -/
protected lemma ext (φ : a ⟶ b) [IsStronglyCartesian p f φ] {R' : 𝒮} {a' : 𝒳} (g : R' ⟶ R)
    {ψ ψ' : a' ⟶ a} [IsHomLift p g ψ] [IsHomLift p g ψ'] (h : ψ ≫ φ = ψ' ≫ φ) : ψ = ψ' := by
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
    inst✝² : p.IsStronglyCartesian f φ
    R' : 𝒮
    a' : 𝒳
    g : Quiver.Hom R' R
    ψ ψ' : Quiver.Hom a' a
    inst✝¹ : p.IsHomLift g ψ
    inst✝ : p.IsHomLift g ψ'
    h : Eq (CategoryTheory.CategoryStruct.comp ψ φ) (CategoryTheory.CategoryStruct …
    ⊢ Eq ψ ψ'
  -/
  rw [map_uniq p f φ (g := g) rfl (ψ ≫ φ) ψ rfl, map_uniq p f φ (g := g) rfl (ψ ≫ φ) ψ' h.symm]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_self : map p f φ (id_comp f).symm φ = 𝟙 a := by
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
    inst✝ : p.IsStronglyCartesian f φ
    ⊢ Eq (CategoryTheory.Functor.IsStronglyCartesian.map p f φ ⋯ φ) (CategoryTheor …
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
    inst✝ : p.IsStronglyCartesian (p.map φ) φ
    ⊢ Eq (CategoryTheory.CategoryStruct.id a✝) (CategoryTheory.Functor.IsStronglyC …
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
    inst✝ : p.IsStronglyCartesian (p.map φ) φ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id a✝) …
  -/
  simp only [id_comp]
  /-
    🎉 no goals
  -/


/-- When its possible to compare the two, the composition of two `IsStronglyCartesian.map` will also
be given by a `IsStronglyCartesian.map`. In other words, given diagrams
```
a''         a'        a --φ--> b
|           |         |        |
v           v         v        v
R'' --g'--> R' --g--> R --f--> S
```
and
```
a' --φ'--> b
|          |
v          v
R' --f'--> S
```
and
```
a'' --φ''--> b
|            |
v            v
R'' --f''--> S
```
such that `φ` and `φ'` are strongly cartesian morphisms, and such that `f' = g ≫ f` and
`f'' = g' ≫ f'`. Then composing the induced map from `a'' ⟶ a'` with the induced map from
`a' ⟶ a` gives the induced map from `a'' ⟶ a`. -/
@[reassoc (attr := simp)]
lemma map_comp_map {R' R'' : 𝒮} {a' a'' : 𝒳} {f' : R' ⟶ S} {f'' : R'' ⟶ S} {g : R' ⟶ R}
    {g' : R'' ⟶ R'} (H : f' = g ≫ f) (H' : f'' = g' ≫ f') (φ' : a' ⟶ b) (φ'' : a'' ⟶ b)
    [IsStronglyCartesian p f' φ'] [IsHomLift p f'' φ''] :
    map p f' φ' H' φ'' ≫ map p f φ H φ' =
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
                                              inst✝² : p.IsStronglyCartesian f φ
                                              R' R'' : 𝒮
                                              a' a'' : 𝒳
                                              f' : Quiver.Hom R' S
                                              f'' : Quiver.Hom R'' S
                                              g : Quiver.Hom R' R
                                              g' : Quiver.Hom R'' R'
                                              H : Eq f' (CategoryTheory.CategoryStruct.comp g f)
                                              H' : Eq f'' (CategoryTheory.CategoryStruct.comp g' f')
                                              φ' : Quiver.Hom a' b
                                              φ'' : Quiver.Hom a'' b
                                              inst✝¹ : p.IsStronglyCartesian f' φ'
                                              inst✝ : p.IsHomLift f'' φ''
                                              ⊢ Eq f'' (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
                                            -/
      map p f φ (show f'' = (g' ≫ g) ≫ f by rwa [assoc, ← H]) φ'' := by
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
    inst✝² : p.IsStronglyCartesian f φ
    R' R'' : 𝒮
    a' a'' : 𝒳
    f' : Quiver.Hom R' S
    f'' : Quiver.Hom R'' S
    g : Quiver.Hom R' R
    g' : Quiver.Hom R'' R'
    H : Eq f' (CategoryTheory.CategoryStruct.comp g f)
    H' : Eq f'' (CategoryTheory.CategoryStruct.comp g' f')
    φ' : Quiver.Hom a' b
    φ'' : Quiver.Hom a'' b
    inst✝¹ : p.IsStronglyCartesian f' φ'
    inst✝ : p.IsHomLift f'' φ''
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsStronglyCar …
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
    inst✝² : p.IsStronglyCartesian f φ
    R' R'' : 𝒮
    a' a'' : 𝒳
    f' : Quiver.Hom R' S
    f'' : Quiver.Hom R'' S
    g : Quiver.Hom R' R
    g' : Quiver.Hom R'' R'
    H : Eq f' (CategoryTheory.CategoryStruct.comp g f)
    H' : Eq f'' (CategoryTheory.CategoryStruct.comp g' f')
    φ' : Quiver.Hom a' b
    φ'' : Quiver.Hom a'' b
    inst✝¹ : p.IsStronglyCartesian f' φ'
    inst✝ : p.IsHomLift f'' φ''
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [assoc, fac]
  /-
    🎉 no goals
  -/


/-- Given two strongly cartesian morphisms `φ`, `ψ` as follows
```
a --φ--> b --ψ--> c
|        |        |
v        v        v
R --f--> S --g--> T
```
Then the composite `φ ≫ ψ` is also strongly cartesian. -/
instance comp [IsStronglyCartesian p f φ] [IsStronglyCartesian p g ψ] :
    IsStronglyCartesian p (f ≫ g) (φ ≫ ψ) where
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
      inst✝¹ : p.IsStronglyCartesian f φ
      inst✝ : p.IsStronglyCartesian g ψ
      ⊢ ∀ {a' : 𝒳} (g_1 : Quiver.Hom (p.obj a') R) (φ' : Quiver.Hom a' c) [inst : p. …
    -/
    intro a' h τ hτ
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
      inst✝¹ : p.IsStronglyCartesian f φ
      inst✝ : p.IsStronglyCartesian g ψ
      a' : 𝒳
      h : Quiver.Hom (p.obj a') R
      τ : Quiver.Hom a' c
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
      ⊢ ExistsUnique fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory.CategoryStru …
    -/
    use map p f φ (f' := h ≫ f) rfl (map p g ψ (assoc h f g).symm τ)
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
      inst✝¹ : p.IsStronglyCartesian f φ
      inst✝ : p.IsStronglyCartesian g ψ
      a' : 𝒳
      h : Quiver.Hom (p.obj a') R
      τ : Quiver.Hom a' c
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
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
        inst✝¹ : p.IsStronglyCartesian f φ
        inst✝ : p.IsStronglyCartesian g ψ
        a' : 𝒳
        h : Quiver.Hom (p.obj a') R
        τ : Quiver.Hom a' c
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsStronglyCar …
      -/
    · rw [← assoc, fac, fac]
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
        inst✝¹ : p.IsStronglyCartesian f φ
        inst✝ : p.IsStronglyCartesian g ψ
        a' : 𝒳
        h : Quiver.Hom (p.obj a') R
        τ : Quiver.Hom a' c
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
        ⊢ ∀ (y : Quiver.Hom a' a), (fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory …
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
        inst✝¹ : p.IsStronglyCartesian f φ
        inst✝ : p.IsStronglyCartesian g ψ
        a' : 𝒳
        h : Quiver.Hom (p.obj a') R
        τ : Quiver.Hom a' c
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
        π' : Quiver.Hom a' a
        hπ'₁ : p.IsHomLift h π'
        hπ'₂ : Eq (CategoryTheory.CategoryStruct.comp π' (CategoryTheory.CategoryStruc …
        ⊢ Eq π' (CategoryTheory.Functor.IsStronglyCartesian.map p f φ ⋯ (CategoryTheor …
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
        inst✝¹ : p.IsStronglyCartesian f φ
        inst✝ : p.IsStronglyCartesian g ψ
        a' : 𝒳
        h : Quiver.Hom (p.obj a') R
        τ : Quiver.Hom a' c
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
        π' : Quiver.Hom a' a
        hπ'₁ : p.IsHomLift h π'
        hπ'₂ : Eq (CategoryTheory.CategoryStruct.comp π' (CategoryTheory.CategoryStruc …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp π' φ) (CategoryTheory.Functor.IsStron …
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
        inst✝¹ : p.IsStronglyCartesian f φ
        inst✝ : p.IsStronglyCartesian g ψ
        a' : 𝒳
        h : Quiver.Hom (p.obj a') R
        τ : Quiver.Hom a' c
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
        π' : Quiver.Hom a' a
        hπ'₁ : p.IsHomLift h π'
        hπ'₂ : Eq (CategoryTheory.CategoryStruct.comp π' (CategoryTheory.CategoryStruc …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp π …
      -/
      simp only [assoc, hπ'₂]
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
such that `φ ≫ ψ` and `ψ` are strongly cartesian, then so is `φ`. -/
protected lemma of_comp [IsStronglyCartesian p g ψ] [IsStronglyCartesian p (f ≫ g) (φ ≫ ψ)]
    [IsHomLift p f φ] : IsStronglyCartesian p f φ where
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
      inst✝² : p.IsStronglyCartesian g ψ
      inst✝¹ : p.IsStronglyCartesian (CategoryTheory.CategoryStruct.comp f g) (Categ …
      inst✝ : p.IsHomLift f φ
      ⊢ ∀ {a' : 𝒳} (g : Quiver.Hom (p.obj a') R) (φ' : Quiver.Hom a' b) [inst : p.Is …
    -/
    intro a' h τ hτ
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
      inst✝² : p.IsStronglyCartesian g ψ
      inst✝¹ : p.IsStronglyCartesian (CategoryTheory.CategoryStruct.comp f g) (Categ …
      inst✝ : p.IsHomLift f φ
      a' : 𝒳
      h : Quiver.Hom (p.obj a') R
      τ : Quiver.Hom a' b
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h f) τ
      ⊢ ExistsUnique fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory.CategoryStru …
    -/
    have h₁ : IsHomLift p (h ≫ f ≫ g) (τ ≫ ψ) := by simpa using IsHomLift.comp p (h ≫ f) _ τ ψ
    /- We get a morphism `π : a' ⟶ a` such that `π ≫ φ ≫ ψ = τ ≫ ψ` from the universal property
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
      inst✝² : p.IsStronglyCartesian g ψ
      inst✝¹ : p.IsStronglyCartesian (CategoryTheory.CategoryStruct.comp f g) (Categ …
      inst✝ : p.IsHomLift f φ
      a' : 𝒳
      h : Quiver.Hom (p.obj a') R
      τ : Quiver.Hom a' b
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h f) τ
      h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
      ⊢ ExistsUnique fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory.CategoryStru …
    -/
    use map p (f ≫ g) (φ ≫ ψ) (f' := h ≫ f ≫ g) rfl (τ ≫ ψ)
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
      inst✝² : p.IsStronglyCartesian g ψ
      inst✝¹ : p.IsStronglyCartesian (CategoryTheory.CategoryStruct.comp f g) (Categ …
      inst✝ : p.IsHomLift f φ
      a' : 𝒳
      h : Quiver.Hom (p.obj a') R
      τ : Quiver.Hom a' b
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h f) τ
      h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
      ⊢ And ((fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory.CategoryStruct.comp …
    -/
    refine ⟨⟨inferInstance, ?_⟩, ?_⟩
    /- The fact that `π ≫ φ = τ` follows from `π ≫ φ ≫ ψ = τ ≫ ψ` and the universal property of
    `ψ`. -/
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
        inst✝² : p.IsStronglyCartesian g ψ
        inst✝¹ : p.IsStronglyCartesian (CategoryTheory.CategoryStruct.comp f g) (Categ …
        inst✝ : p.IsHomLift f φ
        a' : 𝒳
        h : Quiver.Hom (p.obj a') R
        τ : Quiver.Hom a' b
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h f) τ
        h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsStronglyCar …
      -/
    · apply IsStronglyCartesian.ext p g ψ (h ≫ f) (by simp)
      /-
        🎉 no goals
      -/
    -- Finally, the uniqueness of `π` comes from the universal property of `φ ≫ ψ`.
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
        inst✝² : p.IsStronglyCartesian g ψ
        inst✝¹ : p.IsStronglyCartesian (CategoryTheory.CategoryStruct.comp f g) (Categ …
        inst✝ : p.IsHomLift f φ
        a' : 𝒳
        h : Quiver.Hom (p.obj a') R
        τ : Quiver.Hom a' b
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h f) τ
        h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
        ⊢ ∀ (y : Quiver.Hom a' a), (fun χ => And (p.IsHomLift h χ) (Eq (CategoryTheory …
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
        inst✝² : p.IsStronglyCartesian g ψ
        inst✝¹ : p.IsStronglyCartesian (CategoryTheory.CategoryStruct.comp f g) (Categ …
        inst✝ : p.IsHomLift f φ
        a' : 𝒳
        h : Quiver.Hom (p.obj a') R
        τ : Quiver.Hom a' b
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h f) τ
        h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
        π' : Quiver.Hom a' a
        hπ'₁ : p.IsHomLift h π'
        hπ'₂ : Eq (CategoryTheory.CategoryStruct.comp π' φ) τ
        ⊢ Eq π' (CategoryTheory.Functor.IsStronglyCartesian.map p (CategoryTheory.Cate …
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
        inst✝² : p.IsStronglyCartesian g ψ
        inst✝¹ : p.IsStronglyCartesian (CategoryTheory.CategoryStruct.comp f g) (Categ …
        inst✝ : p.IsHomLift f φ
        a' : 𝒳
        h : Quiver.Hom (p.obj a') R
        τ : Quiver.Hom a' b
        hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h f) τ
        h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Categor …
        π' : Quiver.Hom a' a
        hπ'₁ : p.IsHomLift h π'
        hπ'₂ : Eq (CategoryTheory.CategoryStruct.comp π' φ) τ
        ⊢ Eq (CategoryTheory.CategoryStruct.comp π' (CategoryTheory.CategoryStruct.com …
      -/
      simp [hπ'₂.symm]
      /-
        🎉 no goals
      -/


instance of_iso (φ : a ≅ b) [IsHomLift p f φ.hom] : IsStronglyCartesian p f φ.hom where
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
      ⊢ ∀ {a' : 𝒳} (g : Quiver.Hom (p.obj a') R) (φ' : Quiver.Hom a' b) [inst : p.Is …
    -/
    intro a' g τ hτ
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
      a' : 𝒳
      g : Quiver.Hom (p.obj a') R
      τ : Quiver.Hom a' b
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) τ
      ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
    -/
    use τ ≫ φ.inv
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
      a' : 𝒳
      g : Quiver.Hom (p.obj a') R
      τ : Quiver.Hom a' b
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) τ
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
      a' : 𝒳
      g : Quiver.Hom (p.obj a') R
      τ : Quiver.Hom a' b
      hτ : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) τ
      ⊢ (fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStruct.comp χ φ. …
    -/
    simpa using (IsHomLift.comp p (g ≫ f) (isoOfIsoLift p f φ).inv τ φ.inv)
    /-
      🎉 no goals
    -/


instance of_isIso (φ : a ⟶ b) [IsHomLift p f φ] [IsIso φ] : IsStronglyCartesian p f φ :=
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
  @IsStronglyCartesian.of_iso _ _ _ _ p _ _ _ _ f (asIso φ) (by aesop)
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- A strongly cartesian morphism lying over an isomorphism is an isomorphism. -/
lemma isIso_of_base_isIso (φ : a ⟶ b) [IsStronglyCartesian p f φ] [IsIso f] : IsIso φ := by
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
    inst✝¹ : p.IsStronglyCartesian f φ
    inst✝ : CategoryTheory.IsIso f
    ⊢ CategoryTheory.IsIso φ
  -/
  subst_hom_lift p f φ; clear a b R S
  -- Let `φ` be the morphism induced by applying universal property to `𝟙 b` lying over `f⁻¹ ≫ f`.
  /-
    case map
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCartesian (p.map φ) φ
    inst✝ : CategoryTheory.IsIso (p.map φ)
    ⊢ CategoryTheory.IsIso φ
  -/
  let φ' := map p (p.map φ) φ (IsIso.inv_hom_id (p.map φ)).symm (𝟙 b)
  /-
    case map
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCartesian (p.map φ) φ
    inst✝ : CategoryTheory.IsIso (p.map φ)
    φ' : Quiver.Hom b a := CategoryTheory.Functor.IsStronglyCartesian.map p (p.map …
    ⊢ CategoryTheory.IsIso φ
  -/
  use φ'
  -- `φ' ≫ φ = 𝟙 b` follows immediately from the universal property.
  /-
    case h
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCartesian (p.map φ) φ
    inst✝ : CategoryTheory.IsIso (p.map φ)
    φ' : Quiver.Hom b a := CategoryTheory.Functor.IsStronglyCartesian.map p (p.map …
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp φ φ') (CategoryTheory.CategorySt …
  -/
  have inv_hom : φ' ≫ φ = 𝟙 b := fac p (p.map φ) φ _ (𝟙 b)
  /-
    case h
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCartesian (p.map φ) φ
    inst✝ : CategoryTheory.IsIso (p.map φ)
    φ' : Quiver.Hom b a := CategoryTheory.Functor.IsStronglyCartesian.map p (p.map …
    inv_hom : Eq (CategoryTheory.CategoryStruct.comp φ' φ) (CategoryTheory.Categor …
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp φ φ') (CategoryTheory.CategorySt …
  -/
  refine ⟨?_, inv_hom⟩
  -- We will now show that `φ ≫ φ' = 𝟙 a` by showing that `(φ ≫ φ') ≫ φ = 𝟙 a ≫ φ`.
  have h₁ : IsHomLift p (𝟙 (p.obj a)) (φ  ≫ φ') := by
    rw [← IsIso.hom_inv_id (p.map φ)]
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
    inst✝¹ : p.IsStronglyCartesian (p.map φ) φ
    inst✝ : CategoryTheory.IsIso (p.map φ)
    φ' : Quiver.Hom b a := CategoryTheory.Functor.IsStronglyCartesian.map p (p.map …
    inv_hom : Eq (CategoryTheory.CategoryStruct.comp φ' φ) (CategoryTheory.Categor …
    h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id (p.obj a)) (CategoryTheory. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ φ') (CategoryTheory.CategoryStruct. …
  -/
  apply IsStronglyCartesian.ext p (p.map φ) φ (𝟙 (p.obj a))
  /-
    case h
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝¹ : p.IsStronglyCartesian (p.map φ) φ
    inst✝ : CategoryTheory.IsIso (p.map φ)
    φ' : Quiver.Hom b a := CategoryTheory.Functor.IsStronglyCartesian.map p (p.map …
    inv_hom : Eq (CategoryTheory.CategoryStruct.comp φ' φ) (CategoryTheory.Categor …
    h₁ : p.IsHomLift (CategoryTheory.CategoryStruct.id (p.obj a)) (CategoryTheory. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp φ …
  -/
  simp only [assoc, inv_hom, comp_id, id_comp]
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism between the domains of two strongly cartesian morphisms lying over
isomorphic objects. -/
@[simps]
noncomputable def domainIsoOfBaseIso (h : f' = g.hom ≫ f) (φ : a ⟶ b) (φ' : a' ⟶ b)
    [IsStronglyCartesian p f φ] [IsStronglyCartesian p f' φ'] : a' ≅ a where
  hom := map p f φ h φ'
  inv :=
    haveI : p.IsHomLift ((fun x ↦ g.inv ≫ x) (g.hom ≫ f)) φ := by
      /-
        𝒮 : Type u₁
        𝒳 : Type u₂
        inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
        inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
        p : CategoryTheory.Functor 𝒳 𝒮
        R R' S : 𝒮
        a a' b : 𝒳
        f : Quiver.Hom R S
        f' : Quiver.Hom R' S
        g : CategoryTheory.Iso R' R
        h : Eq f' (CategoryTheory.CategoryStruct.comp g.hom f)
        φ : Quiver.Hom a b
        φ' : Quiver.Hom a' b
        inst✝¹ : p.IsStronglyCartesian f φ
        inst✝ : p.IsStronglyCartesian f' φ'
        ⊢ p.IsHomLift ((fun x => CategoryTheory.CategoryStruct.comp g.inv x) (Category …
      -/
      simpa using IsCartesian.toIsHomLift
      /-
        🎉 no goals
      -/
    map p f' φ' (congrArg (g.inv ≫ ·) h.symm) φ


instance domainUniqueUpToIso_inv_isHomLift (h : f' = g.hom ≫ f) (φ : a ⟶ b) (φ' : a' ⟶ b)
    [IsStronglyCartesian p f φ] [IsStronglyCartesian p f' φ'] :
    IsHomLift p g.hom (domainIsoOfBaseIso p h φ φ').hom :=
  domainIsoOfBaseIso_hom p h φ φ' ▸ IsStronglyCartesian.map_isHomLift p f φ h φ'


instance domainUniqueUpToIso_hom_isHomLift (h : f' = g.hom ≫ f) (φ : a ⟶ b) (φ' : a' ⟶ b)
    [IsStronglyCartesian p f φ] [IsStronglyCartesian p f' φ'] :
    IsHomLift p g.inv (domainIsoOfBaseIso p h φ φ').inv := by
  haveI : p.IsHomLift ((fun x ↦ g.inv ≫ x) (g.hom ≫ f)) φ := by
    simpa using IsCartesian.toIsHomLift
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    R R' S : 𝒮
    a a' b : 𝒳
    f : Quiver.Hom R S
    f' : Quiver.Hom R' S
    g : CategoryTheory.Iso R' R
    h : Eq f' (CategoryTheory.CategoryStruct.comp g.hom f)
    φ : Quiver.Hom a b
    φ' : Quiver.Hom a' b
    inst✝¹ : p.IsStronglyCartesian f φ
    inst✝ : p.IsStronglyCartesian f' φ'
    this : p.IsHomLift ((fun x => CategoryTheory.CategoryStruct.comp g.inv x) (Cat …
    ⊢ p.IsHomLift g.inv (CategoryTheory.Functor.IsStronglyCartesian.domainIsoOfBas …
  -/
  simpa using IsStronglyCartesian.map_isHomLift p f' φ' (congrArg (g.inv ≫ ·) h.symm) φ
  /-
    🎉 no goals
  -/


