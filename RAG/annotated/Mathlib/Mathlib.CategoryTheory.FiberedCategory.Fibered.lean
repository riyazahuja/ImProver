/-- Definition of a prefibered category.

See SGA 1 VI.6.1. -/
class Functor.IsPreFibered (p : 𝒳 ⥤ 𝒮) : Prop where
  exists_isCartesian' {a : 𝒳} {R : 𝒮} (f : R ⟶ p.obj a) : ∃ (b : 𝒳) (φ : b ⟶ a), IsCartesian p f φ


protected lemma IsPreFibered.exists_isCartesian (p : 𝒳 ⥤ 𝒮) [p.IsPreFibered] {a : 𝒳} {R S : 𝒮}
    (ha : p.obj a = S) (f : R ⟶ S) : ∃ (b : 𝒳) (φ : b ⟶ a), IsCartesian p f φ := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    inst✝ : p.IsPreFibered
    a : 𝒳
    R S : 𝒮
    ha : Eq (p.obj a) S
    f : Quiver.Hom R S
    ⊢ Exists fun b => Exists fun φ => p.IsCartesian f φ
  -/
  subst ha; exact IsPreFibered.exists_isCartesian' f
            /-
              🎉 no goals
            -/


/-- Definition of a fibered category.

See SGA 1 VI.6.1. -/
class Functor.IsFibered (p : 𝒳 ⥤ 𝒮) extends IsPreFibered p : Prop where
  comp {R S T : 𝒮} (f : R ⟶ S) (g : S ⟶ T) {a b c : 𝒳} (φ : a ⟶ b) (ψ : b ⟶ c)
    [IsCartesian p f φ] [IsCartesian p g ψ] : IsCartesian p (f ≫ g) (φ ≫ ψ)


instance (p : 𝒳 ⥤ 𝒮) [p.IsFibered] {R S T : 𝒮} (f : R ⟶ S) (g : S ⟶ T) {a b c : 𝒳} (φ : a ⟶ b)
    (ψ : b ⟶ c) [IsCartesian p f φ] [IsCartesian p g ψ] : IsCartesian p (f ≫ g) (φ ≫ ψ) :=
  IsFibered.comp f g φ ψ


/-- Given a fibered category `p : 𝒳 ⥤ 𝒫`, a morphism `f : R ⟶ S` and an object `a` lying over `S`,
then `pullbackObj` is the domain of some choice of a cartesian morphism lying over `f` with
codomain `a`. -/
noncomputable def pullbackObj : 𝒳 :=
  Classical.choose (IsPreFibered.exists_isCartesian p ha f)


/-- Given a fibered category `p : 𝒳 ⥤ 𝒫`, a morphism `f : R ⟶ S` and an object `a` lying over `S`,
then `pullbackMap` is a choice of a cartesian morphism lying over `f` with codomain `a`. -/
noncomputable def pullbackMap : pullbackObj ha f ⟶ a :=
  Classical.choose (Classical.choose_spec (IsPreFibered.exists_isCartesian p ha f))


instance pullbackMap.IsCartesian : IsCartesian p f (pullbackMap ha f) :=
  Classical.choose_spec (Classical.choose_spec (IsPreFibered.exists_isCartesian p ha f))


lemma pullbackObj_proj : p.obj (pullbackObj ha f) = R :=
  domain_eq p f (pullbackMap ha f)


/-- In a fibered category, any cartesian morphism is strongly cartesian. -/
instance isStronglyCartesian_of_isCartesian (p : 𝒳 ⥤ 𝒮) [p.IsFibered] {R S : 𝒮} (f : R ⟶ S)
    {a b : 𝒳} (φ : a ⟶ b) [p.IsCartesian f φ] : p.IsStronglyCartesian f φ where
  universal_property' g φ' hφ' := by
    -- Let `ψ` be a cartesian arrow lying over `g`
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      inst✝¹ : p.IsFibered
      R S : 𝒮
      f : Quiver.Hom R S
      a b : 𝒳
      φ : Quiver.Hom a b
      inst✝ : p.IsCartesian f φ
      a'✝ : 𝒳
      g : Quiver.Hom (p.obj a'✝) R
      φ' : Quiver.Hom a'✝ b
      hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) φ'
      ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
    -/
    let ψ := pullbackMap (domain_eq p f φ) g
    -- Let `τ` be the map induced by the universal property of `ψ ≫ φ`.
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      inst✝¹ : p.IsFibered
      R S : 𝒮
      f : Quiver.Hom R S
      a b : 𝒳
      φ : Quiver.Hom a b
      inst✝ : p.IsCartesian f φ
      a'✝ : 𝒳
      g : Quiver.Hom (p.obj a'✝) R
      φ' : Quiver.Hom a'✝ b
      hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) φ'
      ψ : Quiver.Hom (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) a := Cate …
      ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
    -/
    let τ := IsCartesian.map p (g ≫ f) (ψ ≫ φ) φ'
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      inst✝¹ : p.IsFibered
      R S : 𝒮
      f : Quiver.Hom R S
      a b : 𝒳
      φ : Quiver.Hom a b
      inst✝ : p.IsCartesian f φ
      a'✝ : 𝒳
      g : Quiver.Hom (p.obj a'✝) R
      φ' : Quiver.Hom a'✝ b
      hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) φ'
      ψ : Quiver.Hom (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) a := Cate …
      τ : Quiver.Hom a'✝ (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) := Ca …
      ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
    -/
    use τ ≫ ψ
    -- It is easily verified that `τ ≫ ψ` lifts `g` and `τ ≫ ψ ≫ φ = φ'`
    /-
      case h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      inst✝¹ : p.IsFibered
      R S : 𝒮
      f : Quiver.Hom R S
      a b : 𝒳
      φ : Quiver.Hom a b
      inst✝ : p.IsCartesian f φ
      a'✝ : 𝒳
      g : Quiver.Hom (p.obj a'✝) R
      φ' : Quiver.Hom a'✝ b
      hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) φ'
      ψ : Quiver.Hom (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) a := Cate …
      τ : Quiver.Hom a'✝ (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) := Ca …
      ⊢ And ((fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStruct.comp …
    -/
    refine ⟨⟨inferInstance, by simp only [assoc, IsCartesian.fac, τ]⟩, ?_⟩
    -- It remains to check that `τ ≫ ψ` is unique.
    -- So fix another lift `π` of `g` satisfying `π ≫ φ = φ'`.
    /-
      case h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      inst✝¹ : p.IsFibered
      R S : 𝒮
      f : Quiver.Hom R S
      a b : 𝒳
      φ : Quiver.Hom a b
      inst✝ : p.IsCartesian f φ
      a'✝ : 𝒳
      g : Quiver.Hom (p.obj a'✝) R
      φ' : Quiver.Hom a'✝ b
      hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) φ'
      ψ : Quiver.Hom (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) a := Cate …
      τ : Quiver.Hom a'✝ (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) := Ca …
      ⊢ ∀ (y : Quiver.Hom a'✝ a), (fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheor …
    -/
    intro π ⟨hπ, hπ_comp⟩
    -- Write `π` as `π = τ' ≫ ψ` for some `τ'` induced by the universal property of `ψ`.
    /-
      case h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      inst✝¹ : p.IsFibered
      R S : 𝒮
      f : Quiver.Hom R S
      a b : 𝒳
      φ : Quiver.Hom a b
      inst✝ : p.IsCartesian f φ
      a'✝ : 𝒳
      g : Quiver.Hom (p.obj a'✝) R
      φ' : Quiver.Hom a'✝ b
      hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) φ'
      ψ : Quiver.Hom (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) a := Cate …
      τ : Quiver.Hom a'✝ (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) := Ca …
      π : Quiver.Hom a'✝ a
      hπ : p.IsHomLift g π
      hπ_comp : Eq (CategoryTheory.CategoryStruct.comp π φ) φ'
      ⊢ Eq π (CategoryTheory.CategoryStruct.comp τ ψ)
    -/
    rw [← fac p g ψ π]
    -- It remains to show that `τ' = τ`. This follows again from the universal property of `ψ`.
    /-
      case h
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      inst✝¹ : p.IsFibered
      R S : 𝒮
      f : Quiver.Hom R S
      a b : 𝒳
      φ : Quiver.Hom a b
      inst✝ : p.IsCartesian f φ
      a'✝ : 𝒳
      g : Quiver.Hom (p.obj a'✝) R
      φ' : Quiver.Hom a'✝ b
      hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) φ'
      ψ : Quiver.Hom (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) a := Cate …
      τ : Quiver.Hom a'✝ (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) := Ca …
      π : Quiver.Hom a'✝ a
      hπ : p.IsHomLift g π
      hπ_comp : Eq (CategoryTheory.CategoryStruct.comp π φ) φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCartesian.m …
    -/
    congr 1
    /-
      case h.e_a
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      inst✝¹ : p.IsFibered
      R S : 𝒮
      f : Quiver.Hom R S
      a b : 𝒳
      φ : Quiver.Hom a b
      inst✝ : p.IsCartesian f φ
      a'✝ : 𝒳
      g : Quiver.Hom (p.obj a'✝) R
      φ' : Quiver.Hom a'✝ b
      hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) φ'
      ψ : Quiver.Hom (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) a := Cate …
      τ : Quiver.Hom a'✝ (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) := Ca …
      π : Quiver.Hom a'✝ a
      hπ : p.IsHomLift g π
      hπ_comp : Eq (CategoryTheory.CategoryStruct.comp π φ) φ'
      ⊢ Eq (CategoryTheory.Functor.IsCartesian.map p g ψ π) τ
    -/
    apply map_uniq
    /-
      case h.e_a.hψ
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝² : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      inst✝¹ : p.IsFibered
      R S : 𝒮
      f : Quiver.Hom R S
      a b : 𝒳
      φ : Quiver.Hom a b
      inst✝ : p.IsCartesian f φ
      a'✝ : 𝒳
      g : Quiver.Hom (p.obj a'✝) R
      φ' : Quiver.Hom a'✝ b
      hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) φ'
      ψ : Quiver.Hom (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) a := Cate …
      τ : Quiver.Hom a'✝ (CategoryTheory.Functor.IsPreFibered.pullbackObj ⋯ g) := Ca …
      π : Quiver.Hom a'✝ a
      hπ : p.IsHomLift g π
      hπ_comp : Eq (CategoryTheory.CategoryStruct.comp π φ) φ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCartesian.m …
    -/
    rwa [← assoc, IsCartesian.fac]
    /-
      🎉 no goals
    -/


/-- In a category which admits strongly cartesian pullbacks, any cartesian morphism is
strongly cartesian. This is a helper-lemma for the fact that admitting strongly cartesian pullbacks
implies being fibered. -/
lemma isStronglyCartesian_of_exists_isCartesian (p : 𝒳 ⥤ 𝒮) (h : ∀ (a : 𝒳) (R : 𝒮)
    (f : R ⟶ p.obj a), ∃ (b : 𝒳) (φ : b ⟶ a), IsStronglyCartesian p f φ) {R S : 𝒮} (f : R ⟶ S)
      {a b : 𝒳} (φ : a ⟶ b) [p.IsCartesian f φ] : p.IsStronglyCartesian f φ := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
    R S : 𝒮
    f : Quiver.Hom R S
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝ : p.IsCartesian f φ
    ⊢ p.IsStronglyCartesian f φ
  -/
  constructor
  /-
    case universal_property'
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
    R S : 𝒮
    f : Quiver.Hom R S
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝ : p.IsCartesian f φ
    ⊢ ∀ {a' : 𝒳} (g : Quiver.Hom (p.obj a') R) (φ' : Quiver.Hom a' b) [inst : p.Is …
  -/
  intro c g φ' hφ'
  /-
    case universal_property'
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
    R S : 𝒮
    f : Quiver.Hom R S
    a b : 𝒳
    φ : Quiver.Hom a b
    inst✝ : p.IsCartesian f φ
    c : 𝒳
    g : Quiver.Hom (p.obj c) R
    φ' : Quiver.Hom c b
    hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g f) φ'
    ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
  -/
  subst_hom_lift p f φ; clear a b R S
  -- Let `ψ` be a cartesian arrow lying over `g`
  /-
    case universal_property'.map
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
    a b : 𝒳
    φ : Quiver.Hom a b
    c : 𝒳
    φ' : Quiver.Hom c b
    g : Quiver.Hom (p.obj c) (p.obj a)
    inst✝ : p.IsCartesian (p.map φ) φ
    hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g (p.map φ)) φ'
    ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
  -/
  obtain ⟨a', ψ, hψ⟩ := h _ _ (p.map φ)
  -- Let `τ' : c ⟶ a'` be the map induced by the universal property of `ψ`
  /-
    case universal_property'.map.intro.intro
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
    a b : 𝒳
    φ : Quiver.Hom a b
    c : 𝒳
    φ' : Quiver.Hom c b
    g : Quiver.Hom (p.obj c) (p.obj a)
    inst✝ : p.IsCartesian (p.map φ) φ
    hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g (p.map φ)) φ'
    a' : 𝒳
    ψ : Quiver.Hom a' b
    hψ : p.IsStronglyCartesian (p.map φ) ψ
    ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
  -/
  let τ' := IsStronglyCartesian.map p (p.map φ) ψ (f':= g ≫ p.map φ) rfl φ'
  -- Let `Φ : a' ≅ a` be natural isomorphism induced between `φ` and `ψ`.
  /-
    case universal_property'.map.intro.intro
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
    a b : 𝒳
    φ : Quiver.Hom a b
    c : 𝒳
    φ' : Quiver.Hom c b
    g : Quiver.Hom (p.obj c) (p.obj a)
    inst✝ : p.IsCartesian (p.map φ) φ
    hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g (p.map φ)) φ'
    a' : 𝒳
    ψ : Quiver.Hom a' b
    hψ : p.IsStronglyCartesian (p.map φ) ψ
    τ' : Quiver.Hom c a' := CategoryTheory.Functor.IsStronglyCartesian.map p (p.ma …
    ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
  -/
  let Φ := domainUniqueUpToIso p (p.map φ) φ ψ
  -- The map induced by `φ` will be `τ' ≫ Φ.hom`
  /-
    case universal_property'.map.intro.intro
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
    a b : 𝒳
    φ : Quiver.Hom a b
    c : 𝒳
    φ' : Quiver.Hom c b
    g : Quiver.Hom (p.obj c) (p.obj a)
    inst✝ : p.IsCartesian (p.map φ) φ
    hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g (p.map φ)) φ'
    a' : 𝒳
    ψ : Quiver.Hom a' b
    hψ : p.IsStronglyCartesian (p.map φ) ψ
    τ' : Quiver.Hom c a' := CategoryTheory.Functor.IsStronglyCartesian.map p (p.ma …
    Φ : CategoryTheory.Iso a' a := CategoryTheory.Functor.IsCartesian.domainUnique …
    ⊢ ExistsUnique fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStru …
  -/
  use τ' ≫ Φ.hom
  -- It is easily verified that `τ' ≫ Φ.hom` lifts `g` and `τ' ≫ Φ.hom ≫ φ = φ'`
  /-
    case h
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
    a b : 𝒳
    φ : Quiver.Hom a b
    c : 𝒳
    φ' : Quiver.Hom c b
    g : Quiver.Hom (p.obj c) (p.obj a)
    inst✝ : p.IsCartesian (p.map φ) φ
    hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g (p.map φ)) φ'
    a' : 𝒳
    ψ : Quiver.Hom a' b
    hψ : p.IsStronglyCartesian (p.map φ) ψ
    τ' : Quiver.Hom c a' := CategoryTheory.Functor.IsStronglyCartesian.map p (p.ma …
    Φ : CategoryTheory.Iso a' a := CategoryTheory.Functor.IsCartesian.domainUnique …
    ⊢ And ((fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory.CategoryStruct.comp …
  -/
  refine ⟨⟨by simp only [Φ]; infer_instance, ?_⟩, ?_⟩
    /-
      case h.refine_1
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
      a b : 𝒳
      φ : Quiver.Hom a b
      c : 𝒳
      φ' : Quiver.Hom c b
      g : Quiver.Hom (p.obj c) (p.obj a)
      inst✝ : p.IsCartesian (p.map φ) φ
      hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g (p.map φ)) φ'
      a' : 𝒳
      ψ : Quiver.Hom a' b
      hψ : p.IsStronglyCartesian (p.map φ) ψ
      τ' : Quiver.Hom c a' := CategoryTheory.Functor.IsStronglyCartesian.map p (p.ma …
      Φ : CategoryTheory.Iso a' a := CategoryTheory.Functor.IsCartesian.domainUnique …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp τ …
    -/
  · simp [τ', Φ, IsStronglyCartesian.map_uniq p (p.map φ) ψ rfl φ']
    /-
      🎉 no goals
    -/
  -- It remains to check that it is unique. This follows from the universal property of `ψ`.
  /-
    case h.refine_2
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
    a b : 𝒳
    φ : Quiver.Hom a b
    c : 𝒳
    φ' : Quiver.Hom c b
    g : Quiver.Hom (p.obj c) (p.obj a)
    inst✝ : p.IsCartesian (p.map φ) φ
    hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g (p.map φ)) φ'
    a' : 𝒳
    ψ : Quiver.Hom a' b
    hψ : p.IsStronglyCartesian (p.map φ) ψ
    τ' : Quiver.Hom c a' := CategoryTheory.Functor.IsStronglyCartesian.map p (p.ma …
    Φ : CategoryTheory.Iso a' a := CategoryTheory.Functor.IsCartesian.domainUnique …
    ⊢ ∀ (y : Quiver.Hom c a), (fun χ => And (p.IsHomLift g χ) (Eq (CategoryTheory. …
  -/
  intro π ⟨hπ, hπ_comp⟩
  /-
    case h.refine_2
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
    a b : 𝒳
    φ : Quiver.Hom a b
    c : 𝒳
    φ' : Quiver.Hom c b
    g : Quiver.Hom (p.obj c) (p.obj a)
    inst✝ : p.IsCartesian (p.map φ) φ
    hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g (p.map φ)) φ'
    a' : 𝒳
    ψ : Quiver.Hom a' b
    hψ : p.IsStronglyCartesian (p.map φ) ψ
    τ' : Quiver.Hom c a' := CategoryTheory.Functor.IsStronglyCartesian.map p (p.ma …
    Φ : CategoryTheory.Iso a' a := CategoryTheory.Functor.IsCartesian.domainUnique …
    π : Quiver.Hom c a
    hπ : p.IsHomLift g π
    hπ_comp : Eq (CategoryTheory.CategoryStruct.comp π φ) φ'
    ⊢ Eq π (CategoryTheory.CategoryStruct.comp τ' Φ.hom)
  -/
  rw [← Iso.comp_inv_eq]
  /-
    case h.refine_2
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
    a b : 𝒳
    φ : Quiver.Hom a b
    c : 𝒳
    φ' : Quiver.Hom c b
    g : Quiver.Hom (p.obj c) (p.obj a)
    inst✝ : p.IsCartesian (p.map φ) φ
    hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g (p.map φ)) φ'
    a' : 𝒳
    ψ : Quiver.Hom a' b
    hψ : p.IsStronglyCartesian (p.map φ) ψ
    τ' : Quiver.Hom c a' := CategoryTheory.Functor.IsStronglyCartesian.map p (p.ma …
    Φ : CategoryTheory.Iso a' a := CategoryTheory.Functor.IsCartesian.domainUnique …
    π : Quiver.Hom c a
    hπ : p.IsHomLift g π
    hπ_comp : Eq (CategoryTheory.CategoryStruct.comp π φ) φ'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp π Φ.inv) τ'
  -/
  apply IsStronglyCartesian.map_uniq p (p.map φ) ψ rfl φ'
  /-
    case h.refine_2.hψ
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} 𝒮
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} 𝒳
    p : CategoryTheory.Functor 𝒳 𝒮
    h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
    a b : 𝒳
    φ : Quiver.Hom a b
    c : 𝒳
    φ' : Quiver.Hom c b
    g : Quiver.Hom (p.obj c) (p.obj a)
    inst✝ : p.IsCartesian (p.map φ) φ
    hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp g (p.map φ)) φ'
    a' : 𝒳
    ψ : Quiver.Hom a' b
    hψ : p.IsStronglyCartesian (p.map φ) ψ
    τ' : Quiver.Hom c a' := CategoryTheory.Functor.IsStronglyCartesian.map p (p.ma …
    Φ : CategoryTheory.Iso a' a := CategoryTheory.Functor.IsCartesian.domainUnique …
    π : Quiver.Hom c a
    hπ : p.IsHomLift g π
    hπ_comp : Eq (CategoryTheory.CategoryStruct.comp π φ) φ'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp π …
  -/
  simp [hπ_comp, Φ]
  /-
    🎉 no goals
  -/


/-- Alternate constructor for `IsFibered`, a functor `p : 𝒳 ⥤ 𝒴` is fibered if any diagram of the
form
```
          a
          -
          |
          v
R --f--> p(a)
```
admits a strongly cartesian lift `b ⟶ a` of `f`. -/
lemma of_exists_isStronglyCartesian {p : 𝒳 ⥤ 𝒮}
    (h : ∀ (a : 𝒳) (R : 𝒮) (f : R ⟶ p.obj a),
      ∃ (b : 𝒳) (φ : b ⟶ a), IsStronglyCartesian p f φ) :
    IsFibered p where
  exists_isCartesian' := by
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝ : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
      ⊢ ∀ {a : 𝒳} {R : 𝒮} (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun φ …
    -/
    intro a R f
    /-
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝ : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
      a : 𝒳
      R : 𝒮
      f : Quiver.Hom R (p.obj a)
      ⊢ Exists fun b => Exists fun φ => p.IsCartesian f φ
    -/
    obtain ⟨b, φ, hφ⟩ := h a R f
    /-
      case intro.intro
      𝒮 : Type u₁
      𝒳 : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} 𝒮
      inst✝ : CategoryTheory.Category.{v₂, u₂} 𝒳
      p : CategoryTheory.Functor 𝒳 𝒮
      h : ∀ (a : 𝒳) (R : 𝒮) (f : Quiver.Hom R (p.obj a)), Exists fun b => Exists fun …
      a : 𝒳
      R : 𝒮
      f : Quiver.Hom R (p.obj a)
      b : 𝒳
      φ : Quiver.Hom b a
      hφ : p.IsStronglyCartesian f φ
      ⊢ Exists fun b => Exists fun φ => p.IsCartesian f φ
    -/
    refine ⟨b, φ, inferInstance⟩
    /-
      🎉 no goals
    -/
  comp := fun R S T f g {a b c} φ ψ _ _ =>
    have : p.IsStronglyCartesian f φ := isStronglyCartesian_of_exists_isCartesian p h _ _
    have : p.IsStronglyCartesian g ψ := isStronglyCartesian_of_exists_isCartesian p h _ _
    inferInstance


/-- Given a diagram
```
                  a
                  -
                  |
                  v
T --g--> R --f--> S
```
we have an isomorphism `T ×_S a ≅ T ×_R (R ×_S a)` -/
noncomputable def pullbackPullbackIso {p : 𝒳 ⥤ 𝒮} [IsFibered p]
    {R S T : 𝒮}  {a : 𝒳} (ha : p.obj a = S) (f : R ⟶ S) (g : T ⟶ R) :
      pullbackObj ha (g ≫ f) ≅ pullbackObj (pullbackObj_proj ha f) g :=
  domainUniqueUpToIso p (g ≫ f) (pullbackMap (pullbackObj_proj ha f) g ≫ pullbackMap ha f)
    (pullbackMap ha (g ≫ f))


