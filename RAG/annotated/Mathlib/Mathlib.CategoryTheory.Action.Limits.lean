instance [HasFiniteProducts V] : HasFiniteProducts (Action V G) where
  out _ :=
    Adjunction.hasLimitsOfShape_of_equivalence (Action.functorCategoryEquivalence _ _).functor


instance [HasFiniteLimits V] : HasFiniteLimits (Action V G) where
  out _ _ _ :=
    Adjunction.hasLimitsOfShape_of_equivalence (Action.functorCategoryEquivalence _ _).functor


instance [HasLimits V] : HasLimits (Action V G) :=
  Adjunction.has_limits_of_equivalence (Action.functorCategoryEquivalence _ _).functor


/-- If `V` has limits of shape `J`, so does `Action V G`. -/
instance hasLimitsOfShape {J : Type w₁} [Category.{w₂} J] [HasLimitsOfShape J V] :
    HasLimitsOfShape J (Action V G) :=
  Adjunction.hasLimitsOfShape_of_equivalence (Action.functorCategoryEquivalence _ _).functor


instance [HasFiniteCoproducts V] : HasFiniteCoproducts (Action V G) where
  out _ :=
    Adjunction.hasColimitsOfShape_of_equivalence (Action.functorCategoryEquivalence _ _).functor


instance [HasFiniteColimits V] : HasFiniteColimits (Action V G) where
  out _ _ _ :=
    Adjunction.hasColimitsOfShape_of_equivalence (Action.functorCategoryEquivalence _ _).functor


instance [HasColimits V] : HasColimits (Action V G) :=
  Adjunction.has_colimits_of_equivalence (Action.functorCategoryEquivalence _ _).functor


/-- If `V` has colimits of shape `J`, so does `Action V G`. -/
instance hasColimitsOfShape {J : Type w₁} [Category.{w₂} J]
    [HasColimitsOfShape J V] : HasColimitsOfShape J (Action V G) :=
  Adjunction.hasColimitsOfShape_of_equivalence (Action.functorCategoryEquivalence _ _).functor


/-- `F : C ⥤ SingleObj G ⥤ V` preserves the limit of some `K : J ⥤ C` if it does
evaluated at `SingleObj.star G`. -/
private lemma SingleObj.preservesLimit (F : C ⥤ SingleObj G ⥤ V)
    {J : Type w₁} [Category.{w₂} J] (K : J ⥤ C)
    (h : PreservesLimit K (F ⋙ (evaluation (SingleObj G) V).obj (SingleObj.star G))) :
    PreservesLimit K F := by
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (CategoryTheory.Functor (CategoryTheory.SingleObj …
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    K : CategoryTheory.Functor J C
    h : CategoryTheory.Limits.PreservesLimit K (F.comp ((CategoryTheory.evaluation …
    ⊢ CategoryTheory.Limits.PreservesLimit K F
  -/
  apply preservesLimit_of_evaluation
  /-
    case H
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (CategoryTheory.Functor (CategoryTheory.SingleObj …
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    K : CategoryTheory.Functor J C
    h : CategoryTheory.Limits.PreservesLimit K (F.comp ((CategoryTheory.evaluation …
    ⊢ ∀ (k : CategoryTheory.SingleObj ↑G), CategoryTheory.Limits.PreservesLimit K  …
  -/
  intro _
  /-
    case H
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (CategoryTheory.Functor (CategoryTheory.SingleObj …
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    K : CategoryTheory.Functor J C
    h : CategoryTheory.Limits.PreservesLimit K (F.comp ((CategoryTheory.evaluation …
    k✝ : CategoryTheory.SingleObj ↑G
    ⊢ CategoryTheory.Limits.PreservesLimit K (F.comp ((CategoryTheory.evaluation ( …
  -/
  exact h
  /-
    🎉 no goals
  -/


/-- `F : C ⥤ Action V G` preserves the limit of some `K : J ⥤ C` if
if it does after postcomposing with the forgetful functor `Action V G ⥤ V`. -/
lemma preservesLimit_of_preserves (F : C ⥤ Action V G) {J : Type w₁}
    [Category.{w₂} J] (K : J ⥤ C)
    (h : PreservesLimit K (F ⋙ Action.forget V G)) : PreservesLimit K F := by
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    K : CategoryTheory.Functor J C
    h : CategoryTheory.Limits.PreservesLimit K (F.comp (Action.forget V G))
    ⊢ CategoryTheory.Limits.PreservesLimit K F
  -/
  let F' : C ⥤ SingleObj G ⥤ V := F ⋙ (Action.functorCategoryEquivalence V G).functor
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    K : CategoryTheory.Functor J C
    h : CategoryTheory.Limits.PreservesLimit K (F.comp (Action.forget V G))
    F' : CategoryTheory.Functor C (CategoryTheory.Functor (CategoryTheory.SingleOb …
    ⊢ CategoryTheory.Limits.PreservesLimit K F
  -/
  have : PreservesLimit K F' := SingleObj.preservesLimit _ _ h
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    K : CategoryTheory.Functor J C
    h : CategoryTheory.Limits.PreservesLimit K (F.comp (Action.forget V G))
    F' : CategoryTheory.Functor C (CategoryTheory.Functor (CategoryTheory.SingleOb …
    this : CategoryTheory.Limits.PreservesLimit K F'
    ⊢ CategoryTheory.Limits.PreservesLimit K F
  -/
  apply preservesLimit_of_reflects_of_preserves F (Action.functorCategoryEquivalence V G).functor
  /-
    🎉 no goals
  -/


/-- `F : C ⥤ Action V G` preserves limits of some shape `J`
if it does after postcomposing with the forgetful functor `Action V G ⥤ V`. -/
lemma preservesLimitsOfShape_of_preserves (F : C ⥤ Action V G) {J : Type w₁}
    [Category.{w₂} J] (h : PreservesLimitsOfShape J (F ⋙ Action.forget V G)) :
    PreservesLimitsOfShape J F := by
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    h : CategoryTheory.Limits.PreservesLimitsOfShape J (F.comp (Action.forget V G))
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J F
  -/
  constructor
  /-
    case preservesLimit
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    h : CategoryTheory.Limits.PreservesLimitsOfShape J (F.comp (Action.forget V G))
    ⊢ autoParam (∀ {K : CategoryTheory.Functor J C}, CategoryTheory.Limits.Preserv …
  -/
  intro K
  /-
    case preservesLimit
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    h : CategoryTheory.Limits.PreservesLimitsOfShape J (F.comp (Action.forget V G))
    K : CategoryTheory.Functor J C
    ⊢ CategoryTheory.Limits.PreservesLimit K F
  -/
  apply Action.preservesLimit_of_preserves
  /-
    case preservesLimit.h
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    h : CategoryTheory.Limits.PreservesLimitsOfShape J (F.comp (Action.forget V G))
    K : CategoryTheory.Functor J C
    ⊢ CategoryTheory.Limits.PreservesLimit K (F.comp (Action.forget V G))
  -/
  exact PreservesLimitsOfShape.preservesLimit
  /-
    🎉 no goals
  -/


/-- `F : C ⥤ Action V G` preserves limits of some size
if it does after postcomposing with the forgetful functor `Action V G ⥤ V`. -/
lemma preservesLimitsOfSize_of_preserves (F : C ⥤ Action V G)
    (h : PreservesLimitsOfSize.{w₂, w₁} (F ⋙ Action.forget V G)) :
    PreservesLimitsOfSize.{w₂, w₁} F := by
  /-
    V : Type (u + 1)
    inst✝¹ : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    h : CategoryTheory.Limits.PreservesLimitsOfSize.{w₂, w₁, t₂, u, t₁, u + 1} (F. …
    ⊢ CategoryTheory.Limits.PreservesLimitsOfSize.{w₂, w₁, t₂, u, t₁, u + 1} F
  -/
  constructor
  /-
    case preservesLimitsOfShape
    V : Type (u + 1)
    inst✝¹ : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    h : CategoryTheory.Limits.PreservesLimitsOfSize.{w₂, w₁, t₂, u, t₁, u + 1} (F. …
    ⊢ autoParam (∀ {J : Type w₁} [inst : CategoryTheory.Category.{w₂, w₁} J], Cate …
  -/
  intro J _
  /-
    case preservesLimitsOfShape
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    h : CategoryTheory.Limits.PreservesLimitsOfSize.{w₂, w₁, t₂, u, t₁, u + 1} (F. …
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J F
  -/
  apply Action.preservesLimitsOfShape_of_preserves
  /-
    case preservesLimitsOfShape.h
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    h : CategoryTheory.Limits.PreservesLimitsOfSize.{w₂, w₁, t₂, u, t₁, u + 1} (F. …
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J (F.comp (Action.forget V G))
  -/
  exact PreservesLimitsOfSize.preservesLimitsOfShape
  /-
    🎉 no goals
  -/


/-- `F : C ⥤ SingleObj G ⥤ V` preserves the colimit of some `K : J ⥤ C` if it does
evaluated at `SingleObj.star G`. -/
private lemma SingleObj.preservesColimit (F : C ⥤ SingleObj G ⥤ V)
    {J : Type w₁} [Category.{w₂} J] (K : J ⥤ C)
    (h : PreservesColimit K (F ⋙ (evaluation (SingleObj G) V).obj (SingleObj.star G))) :
    PreservesColimit K F := by
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (CategoryTheory.Functor (CategoryTheory.SingleObj …
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    K : CategoryTheory.Functor J C
    h : CategoryTheory.Limits.PreservesColimit K (F.comp ((CategoryTheory.evaluati …
    ⊢ CategoryTheory.Limits.PreservesColimit K F
  -/
  apply preservesColimit_of_evaluation
  /-
    case H
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (CategoryTheory.Functor (CategoryTheory.SingleObj …
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    K : CategoryTheory.Functor J C
    h : CategoryTheory.Limits.PreservesColimit K (F.comp ((CategoryTheory.evaluati …
    ⊢ ∀ (k : CategoryTheory.SingleObj ↑G), CategoryTheory.Limits.PreservesColimit  …
  -/
  intro _
  /-
    case H
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (CategoryTheory.Functor (CategoryTheory.SingleObj …
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    K : CategoryTheory.Functor J C
    h : CategoryTheory.Limits.PreservesColimit K (F.comp ((CategoryTheory.evaluati …
    k✝ : CategoryTheory.SingleObj ↑G
    ⊢ CategoryTheory.Limits.PreservesColimit K (F.comp ((CategoryTheory.evaluation …
  -/
  exact h
  /-
    🎉 no goals
  -/


/-- `F : C ⥤ Action V G` preserves the colimit of some `K : J ⥤ C` if
if it does after postcomposing with the forgetful functor `Action V G ⥤ V`. -/
lemma preservesColimit_of_preserves (F : C ⥤ Action V G) {J : Type w₁}
    [Category.{w₂} J] (K : J ⥤ C)
    (h : PreservesColimit K (F ⋙ Action.forget V G)) : PreservesColimit K F := by
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    K : CategoryTheory.Functor J C
    h : CategoryTheory.Limits.PreservesColimit K (F.comp (Action.forget V G))
    ⊢ CategoryTheory.Limits.PreservesColimit K F
  -/
  let F' : C ⥤ SingleObj G ⥤ V := F ⋙ (Action.functorCategoryEquivalence V G).functor
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    K : CategoryTheory.Functor J C
    h : CategoryTheory.Limits.PreservesColimit K (F.comp (Action.forget V G))
    F' : CategoryTheory.Functor C (CategoryTheory.Functor (CategoryTheory.SingleOb …
    ⊢ CategoryTheory.Limits.PreservesColimit K F
  -/
  have : PreservesColimit K F' := SingleObj.preservesColimit _ _ h
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    K : CategoryTheory.Functor J C
    h : CategoryTheory.Limits.PreservesColimit K (F.comp (Action.forget V G))
    F' : CategoryTheory.Functor C (CategoryTheory.Functor (CategoryTheory.SingleOb …
    this : CategoryTheory.Limits.PreservesColimit K F'
    ⊢ CategoryTheory.Limits.PreservesColimit K F
  -/
  apply preservesColimit_of_reflects_of_preserves F (Action.functorCategoryEquivalence V G).functor
  /-
    🎉 no goals
  -/


/-- `F : C ⥤ Action V G` preserves colimits of some shape `J`
if it does after postcomposing with the forgetful functor `Action V G ⥤ V`. -/
lemma preservesColimitsOfShape_of_preserves (F : C ⥤ Action V G) {J : Type w₁}
    [Category.{w₂} J] (h : PreservesColimitsOfShape J (F ⋙ Action.forget V G)) :
    PreservesColimitsOfShape J F := by
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    h : CategoryTheory.Limits.PreservesColimitsOfShape J (F.comp (Action.forget V  …
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J F
  -/
  constructor
  /-
    case preservesColimit
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    h : CategoryTheory.Limits.PreservesColimitsOfShape J (F.comp (Action.forget V  …
    ⊢ autoParam (∀ {K : CategoryTheory.Functor J C}, CategoryTheory.Limits.Preserv …
  -/
  intro K
  /-
    case preservesColimit
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    h : CategoryTheory.Limits.PreservesColimitsOfShape J (F.comp (Action.forget V  …
    K : CategoryTheory.Functor J C
    ⊢ CategoryTheory.Limits.PreservesColimit K F
  -/
  apply Action.preservesColimit_of_preserves
  /-
    case preservesColimit.h
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    h : CategoryTheory.Limits.PreservesColimitsOfShape J (F.comp (Action.forget V  …
    K : CategoryTheory.Functor J C
    ⊢ CategoryTheory.Limits.PreservesColimit K (F.comp (Action.forget V G))
  -/
  exact PreservesColimitsOfShape.preservesColimit
  /-
    🎉 no goals
  -/


/-- `F : C ⥤ Action V G` preserves colimits of some size
if it does after postcomposing with the forgetful functor `Action V G ⥤ V`. -/
lemma preservesColimitsOfSize_of_preserves (F : C ⥤ Action V G)
    (h : PreservesColimitsOfSize.{w₂, w₁} (F ⋙ Action.forget V G)) :
    PreservesColimitsOfSize.{w₂, w₁} F := by
  /-
    V : Type (u + 1)
    inst✝¹ : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    h : CategoryTheory.Limits.PreservesColimitsOfSize.{w₂, w₁, t₂, u, t₁, u + 1} ( …
    ⊢ CategoryTheory.Limits.PreservesColimitsOfSize.{w₂, w₁, t₂, u, t₁, u + 1} F
  -/
  constructor
  /-
    case preservesColimitsOfShape
    V : Type (u + 1)
    inst✝¹ : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    h : CategoryTheory.Limits.PreservesColimitsOfSize.{w₂, w₁, t₂, u, t₁, u + 1} ( …
    ⊢ autoParam (∀ {J : Type w₁} [inst : CategoryTheory.Category.{w₂, w₁} J], Cate …
  -/
  intro J _
  /-
    case preservesColimitsOfShape
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    h : CategoryTheory.Limits.PreservesColimitsOfSize.{w₂, w₁, t₂, u, t₁, u + 1} ( …
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J F
  -/
  apply Action.preservesColimitsOfShape_of_preserves
  /-
    case preservesColimitsOfShape.h
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    C : Type t₁
    inst✝¹ : CategoryTheory.Category.{t₂, t₁} C
    F : CategoryTheory.Functor C (Action V G)
    h : CategoryTheory.Limits.PreservesColimitsOfSize.{w₂, w₁, t₂, u, t₁, u + 1} ( …
    J : Type w₁
    inst✝ : CategoryTheory.Category.{w₂, w₁} J
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J (F.comp (Action.forget V G))
  -/
  exact PreservesColimitsOfSize.preservesColimitsOfShape
  /-
    🎉 no goals
  -/


noncomputable instance {J : Type w₁} [Category.{w₂} J] [HasLimitsOfShape J V] :
    PreservesLimitsOfShape J (Action.forget V G) := by
  show PreservesLimitsOfShape J ((Action.functorCategoryEquivalence V G).functor ⋙
    (evaluation (SingleObj G) V).obj (SingleObj.star G))
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    J : Type w₁
    inst✝¹ : CategoryTheory.Category.{w₂, w₁} J
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J V
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J ((Action.functorCategoryEquiv …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


noncomputable instance {J : Type w₁} [Category.{w₂} J] [HasColimitsOfShape J V] :
    PreservesColimitsOfShape J (Action.forget V G) := by
  show PreservesColimitsOfShape J ((Action.functorCategoryEquivalence V G).functor ⋙
    (evaluation (SingleObj G) V).obj (SingleObj.star G))
  /-
    V : Type (u + 1)
    inst✝² : CategoryTheory.LargeCategory V
    G : MonCat
    J : Type w₁
    inst✝¹ : CategoryTheory.Category.{w₂, w₁} J
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J V
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J ((Action.functorCategoryEqu …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


noncomputable instance [HasFiniteLimits V] : PreservesFiniteLimits (Action.forget V G) := by
  show PreservesFiniteLimits ((Action.functorCategoryEquivalence V G).functor ⋙
    (evaluation (SingleObj G) V).obj (SingleObj.star G))
  have : PreservesFiniteLimits ((evaluation (SingleObj G) V).obj (SingleObj.star G)) := by
    constructor
    intro _ _ _
    infer_instance
  /-
    V : Type (u + 1)
    inst✝¹ : CategoryTheory.LargeCategory V
    G : MonCat
    inst✝ : CategoryTheory.Limits.HasFiniteLimits V
    this : CategoryTheory.Limits.PreservesFiniteLimits ((CategoryTheory.evaluation …
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits ((Action.functorCategoryEquivale …
  -/
  apply comp_preservesFiniteLimits
  /-
    🎉 no goals
  -/


noncomputable instance [HasFiniteColimits V] : PreservesFiniteColimits (Action.forget V G) := by
  show PreservesFiniteColimits ((Action.functorCategoryEquivalence V G).functor ⋙
    (evaluation (SingleObj G) V).obj (SingleObj.star G))
  have : PreservesFiniteColimits ((evaluation (SingleObj G) V).obj (SingleObj.star G)) := by
    constructor
    intro _ _ _
    infer_instance
  /-
    V : Type (u + 1)
    inst✝¹ : CategoryTheory.LargeCategory V
    G : MonCat
    inst✝ : CategoryTheory.Limits.HasFiniteColimits V
    this : CategoryTheory.Limits.PreservesFiniteColimits ((CategoryTheory.evaluati …
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits ((Action.functorCategoryEquiva …
  -/
  apply comp_preservesFiniteColimits
  /-
    🎉 no goals
  -/


instance {J : Type w₁} [Category.{w₂} J] (F : J ⥤ Action V G) :
    ReflectsLimit F (Action.forget V G) where
  reflects h := ⟨by
    /-
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G : MonCat
      J : Type w₁
      inst✝ : CategoryTheory.Category.{w₂, w₁} J
      F : CategoryTheory.Functor J (Action V G)
      c✝ : CategoryTheory.Limits.Cone F
      h : CategoryTheory.Limits.IsLimit ((Action.forget V G).mapCone c✝)
      ⊢ CategoryTheory.Limits.IsLimit c✝
    -/
    apply isLimitOfReflects ((Action.functorCategoryEquivalence V G).functor)
    /-
      case t
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G : MonCat
      J : Type w₁
      inst✝ : CategoryTheory.Category.{w₂, w₁} J
      F : CategoryTheory.Functor J (Action V G)
      c✝ : CategoryTheory.Limits.Cone F
      h : CategoryTheory.Limits.IsLimit ((Action.forget V G).mapCone c✝)
      ⊢ CategoryTheory.Limits.IsLimit ((Action.functorCategoryEquivalence V G).funct …
    -/
    exact evaluationJointlyReflectsLimits _ (fun _ => h)⟩
    /-
      🎉 no goals
    -/


instance {J : Type w₁} [Category.{w₂} J] :
    ReflectsLimitsOfShape J (Action.forget V G) where


instance : ReflectsLimits (Action.forget V G) where


instance {J : Type w₁} [Category.{w₂} J] (F : J ⥤ Action V G) :
    ReflectsColimit F (Action.forget V G) where
  reflects h := ⟨by
    /-
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G : MonCat
      J : Type w₁
      inst✝ : CategoryTheory.Category.{w₂, w₁} J
      F : CategoryTheory.Functor J (Action V G)
      c✝ : CategoryTheory.Limits.Cocone F
      h : CategoryTheory.Limits.IsColimit ((Action.forget V G).mapCocone c✝)
      ⊢ CategoryTheory.Limits.IsColimit c✝
    -/
    apply isColimitOfReflects ((Action.functorCategoryEquivalence V G).functor)
    /-
      case t
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      G : MonCat
      J : Type w₁
      inst✝ : CategoryTheory.Category.{w₂, w₁} J
      F : CategoryTheory.Functor J (Action V G)
      c✝ : CategoryTheory.Limits.Cocone F
      h : CategoryTheory.Limits.IsColimit ((Action.forget V G).mapCocone c✝)
      ⊢ CategoryTheory.Limits.IsColimit ((Action.functorCategoryEquivalence V G).fun …
    -/
    exact evaluationJointlyReflectsColimits _ (fun _ => h)⟩
    /-
      🎉 no goals
    -/


noncomputable instance {J : Type w₁} [Category.{w₂} J] :
    ReflectsColimitsOfShape J (Action.forget V G) where


noncomputable instance : ReflectsColimits (Action.forget V G) where


                                                     /-
                                                       V : Type (u + 1)
                                                       inst✝¹ : CategoryTheory.LargeCategory V
                                                       G : MonCat
                                                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                       X Y : Action V G
                                                       ⊢ ∀ (g : ↑G), Eq (CategoryTheory.CategoryStruct.comp (X.ρ g) 0) (CategoryTheor …
                                                     -/
instance {X Y : Action V G} : Zero (X ⟶ Y) := ⟨0, by aesop_cat⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem zero_hom {X Y : Action V G} : (0 : X ⟶ Y).hom = 0 :=
  rfl


instance : HasZeroMorphisms (Action V G) where


instance forget_preservesZeroMorphisms : Functor.PreservesZeroMorphisms (forget V G) where


instance forget₂_preservesZeroMorphisms [ConcreteCategory V] :
    Functor.PreservesZeroMorphisms (forget₂ (Action V G) V) where


instance functorCategoryEquivalence_preservesZeroMorphisms :
    Functor.PreservesZeroMorphisms (functorCategoryEquivalence V G).functor where


instance {X Y : Action V G} : Add (X ⟶ Y) where
                                /-
                                  V : Type (u + 1)
                                  inst✝¹ : CategoryTheory.LargeCategory V
                                  G : MonCat
                                  inst✝ : CategoryTheory.Preadditive V
                                  X Y : Action V G
                                  f g : Quiver.Hom X Y
                                  ⊢ ∀ (g_1 : ↑G), Eq (CategoryTheory.CategoryStruct.comp (X.ρ g_1) (HAdd.hAdd f. …
                                -/
  add f g := ⟨f.hom + g.hom, by simp [f.comm, g.comm]⟩
                                /-
                                  🎉 no goals
                                -/


instance {X Y : Action V G} : Neg (X ⟶ Y) where
                       /-
                         V : Type (u + 1)
                         inst✝¹ : CategoryTheory.LargeCategory V
                         G : MonCat
                         inst✝ : CategoryTheory.Preadditive V
                         X Y : Action V G
                         f : Quiver.Hom X Y
                         ⊢ ∀ (g : ↑G), Eq (CategoryTheory.CategoryStruct.comp (X.ρ g) (Neg.neg f.hom))  …
                       -/
  neg f := ⟨-f.hom, by simp [f.comm]⟩
                       /-
                         🎉 no goals
                       -/


instance : Preadditive (Action V G) where
  homGroup X Y :=
    { nsmul := nsmulRec
      zsmul := zsmulRec
                     /-
                       V : Type (u + 1)
                       inst✝¹ : CategoryTheory.LargeCategory V
                       G : MonCat
                       inst✝ : CategoryTheory.Preadditive V
                       X Y : Action V G
                       ⊢ ∀ (a : Quiver.Hom X Y), Eq (HAdd.hAdd 0 a) a
                     -/
      zero_add := by intros; ext; exact zero_add _
                      /-
                        V : Type (u + 1)
                        inst✝¹ : CategoryTheory.LargeCategory V
                        G : MonCat
                        inst✝ : CategoryTheory.Preadditive V
                        X Y : Action V G
                        ⊢ ∀ (a b c : Quiver.Hom X Y), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (H …
                      -/
                                  /-
                                    🎉 no goals
                                  -/
                                   /-
                                     🎉 no goals
                                   -/
                     /-
                       V : Type (u + 1)
                       inst✝¹ : CategoryTheory.LargeCategory V
                       G : MonCat
                       inst✝ : CategoryTheory.Preadditive V
                       X Y : Action V G
                       ⊢ ∀ (a : Quiver.Hom X Y), Eq (HAdd.hAdd a 0) a
                     -/
      add_zero := by intros; ext; exact add_zero _
                                  /-
                                    🎉 no goals
                                  -/
      add_assoc := by intros; ext; exact add_assoc _ _ _
                           /-
                             V : Type (u + 1)
                             inst✝¹ : CategoryTheory.LargeCategory V
                             G : MonCat
                             inst✝ : CategoryTheory.Preadditive V
                             X Y : Action V G
                             ⊢ ∀ (a : Quiver.Hom X Y), Eq (HAdd.hAdd (Neg.neg a) a) 0
                           -/
      neg_add_cancel := by intros; ext; exact neg_add_cancel _
                                        /-
                                          🎉 no goals
                                        -/
                     /-
                       V : Type (u + 1)
                       inst✝¹ : CategoryTheory.LargeCategory V
                       G : MonCat
                       inst✝ : CategoryTheory.Preadditive V
                       X Y : Action V G
                       ⊢ ∀ (a b : Quiver.Hom X Y), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
                     -/
      add_comm := by intros; ext; exact add_comm _ _ }
                                  /-
                                    🎉 no goals
                                  -/
                 /-
                   V : Type (u + 1)
                   inst✝¹ : CategoryTheory.LargeCategory V
                   G : MonCat
                   inst✝ : CategoryTheory.Preadditive V
                   ⊢ ∀ (P Q R : Action V G) (f f' : Quiver.Hom P Q) (g : Quiver.Hom Q R), Eq (Cat …
                 -/
  add_comp := by intros; ext; exact Preadditive.add_comp _ _ _ _ _ _
                              /-
                                🎉 no goals
                              -/
                 /-
                   V : Type (u + 1)
                   inst✝¹ : CategoryTheory.LargeCategory V
                   G : MonCat
                   inst✝ : CategoryTheory.Preadditive V
                   ⊢ ∀ (P Q R : Action V G) (f : Quiver.Hom P Q) (g g' : Quiver.Hom Q R), Eq (Cat …
                 -/
  comp_add := by intros; ext; exact Preadditive.comp_add _ _ _ _ _ _
                              /-
                                🎉 no goals
                              -/


instance forget_additive : Functor.Additive (forget V G) where


instance forget₂_additive [ConcreteCategory V] : Functor.Additive (forget₂ (Action V G) V) where


instance functorCategoryEquivalence_additive :
    Functor.Additive (functorCategoryEquivalence V G).functor where


@[simp]
theorem neg_hom {X Y : Action V G} (f : X ⟶ Y) : (-f).hom = -f.hom :=
  rfl


@[simp]
theorem add_hom {X Y : Action V G} (f g : X ⟶ Y) : (f + g).hom = f.hom + g.hom :=
  rfl


@[simp]
theorem sum_hom {X Y : Action V G} {ι : Type*} (f : ι → (X ⟶ Y)) (s : Finset ι) :
    (s.sum f).hom = s.sum fun i => (f i).hom :=
  (forget V G).map_sum f s


instance : Linear R (Action V G) where
  homModule X Y :=
                                        /-
                                          V : Type (u + 1)
                                          inst✝³ : CategoryTheory.LargeCategory V
                                          G : MonCat
                                          inst✝² : CategoryTheory.Preadditive V
                                          R : Type u_1
                                          inst✝¹ : Semiring R
                                          inst✝ : CategoryTheory.Linear R V
                                          X Y : Action V G
                                          r : R
                                          f : Quiver.Hom X Y
                                          ⊢ ∀ (g : ↑G), Eq (CategoryTheory.CategoryStruct.comp (X.ρ g) (HSMul.hSMul r f. …
                                        -/
    { smul := fun r f => ⟨r • f.hom, by simp [f.comm]⟩
                                        /-
                                          🎉 no goals
                                        -/
                     /-
                       V : Type (u + 1)
                       inst✝³ : CategoryTheory.LargeCategory V
                       G : MonCat
                       inst✝² : CategoryTheory.Preadditive V
                       R : Type u_1
                       inst✝¹ : Semiring R
                       inst✝ : CategoryTheory.Linear R V
                       X Y : Action V G
                       ⊢ ∀ (b : Quiver.Hom X Y), Eq (HSMul.hSMul 1 b) b
                     -/
      one_smul := by intros; ext; exact one_smul _ _
                                  /-
                                    🎉 no goals
                                  -/
                      /-
                        V : Type (u + 1)
                        inst✝³ : CategoryTheory.LargeCategory V
                        G : MonCat
                        inst✝² : CategoryTheory.Preadditive V
                        R : Type u_1
                        inst✝¹ : Semiring R
                        inst✝ : CategoryTheory.Linear R V
                        X Y : Action V G
                        ⊢ ∀ (a : R), Eq (HSMul.hSMul a 0) 0
                      -/
      smul_zero := by intros; ext; exact smul_zero _
                                   /-
                                     🎉 no goals
                                   -/
                      /-
                        V : Type (u + 1)
                        inst✝³ : CategoryTheory.LargeCategory V
                        G : MonCat
                        inst✝² : CategoryTheory.Preadditive V
                        R : Type u_1
                        inst✝¹ : Semiring R
                        inst✝ : CategoryTheory.Linear R V
                        X Y : Action V G
                        ⊢ ∀ (x : Quiver.Hom X Y), Eq (HSMul.hSMul 0 x) 0
                      -/
                     /-
                       V : Type (u + 1)
                       inst✝³ : CategoryTheory.LargeCategory V
                       G : MonCat
                       inst✝² : CategoryTheory.Preadditive V
                       R : Type u_1
                       inst✝¹ : Semiring R
                       inst✝ : CategoryTheory.Linear R V
                       X Y : Action V G
                       ⊢ ∀ (x y : R) (b : Quiver.Hom X Y), Eq (HSMul.hSMul (HMul.hMul x y) b) (HSMul. …
                     -/
                     /-
                       V : Type (u + 1)
                       inst✝³ : CategoryTheory.LargeCategory V
                       G : MonCat
                       inst✝² : CategoryTheory.Preadditive V
                       R : Type u_1
                       inst✝¹ : Semiring R
                       inst✝ : CategoryTheory.Linear R V
                       X Y : Action V G
                       ⊢ ∀ (r s : R) (x : Quiver.Hom X Y), Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.h …
                     -/
                                  /-
                                    🎉 no goals
                                  -/
                     /-
                       V : Type (u + 1)
                       inst✝³ : CategoryTheory.LargeCategory V
                       G : MonCat
                       inst✝² : CategoryTheory.Preadditive V
                       R : Type u_1
                       inst✝¹ : Semiring R
                       inst✝ : CategoryTheory.Linear R V
                       X Y : Action V G
                       ⊢ ∀ (a : R) (x y : Quiver.Hom X Y), Eq (HSMul.hSMul a (HAdd.hAdd x y)) (HAdd.h …
                     -/
      zero_smul := by intros; ext; exact zero_smul _ _
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                   /-
                                     🎉 no goals
                                   -/
      add_smul := by intros; ext; exact add_smul _ _ _
      smul_add := by intros; ext; exact smul_add _ _ _
      mul_smul := by intros; ext; exact mul_smul _ _ _ }
                  /-
                    V : Type (u + 1)
                    inst✝³ : CategoryTheory.LargeCategory V
                    G : MonCat
                    inst✝² : CategoryTheory.Preadditive V
                    R : Type u_1
                    inst✝¹ : Semiring R
                    inst✝ : CategoryTheory.Linear R V
                    ⊢ ∀ (X Y Z : Action V G) (r : R) (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq …
                  -/
  smul_comp := by intros; ext; exact Linear.smul_comp _ _ _ _ _ _
                               /-
                                 🎉 no goals
                               -/
                  /-
                    V : Type (u + 1)
                    inst✝³ : CategoryTheory.LargeCategory V
                    G : MonCat
                    inst✝² : CategoryTheory.Preadditive V
                    R : Type u_1
                    inst✝¹ : Semiring R
                    inst✝ : CategoryTheory.Linear R V
                    ⊢ ∀ (X Y Z : Action V G) (f : Quiver.Hom X Y) (r : R) (g : Quiver.Hom Y Z), Eq …
                  -/
  comp_smul := by intros; ext; exact Linear.comp_smul _ _ _ _ _ _
                               /-
                                 🎉 no goals
                               -/


instance forget_linear : Functor.Linear R (forget V G) where


instance forget₂_linear [ConcreteCategory V] : Functor.Linear R (forget₂ (Action V G) V) where


instance functorCategoryEquivalence_linear :
    Functor.Linear R (functorCategoryEquivalence V G).functor where


@[simp]
theorem smul_hom {X Y : Action V G} (r : R) (f : X ⟶ Y) : (r • f).hom = r • f.hom :=
  rfl


instance res_additive : (res V f).Additive where


instance res_linear : (res V f).Linear R where


/-- Auxiliary construction for the `Abelian (Action V G)` instance. -/
def abelianAux : Action V G ≌ ULift.{u} (SingleObj G) ⥤ V :=
  (functorCategoryEquivalence V G).trans (Equivalence.congrLeft ULift.equivalence)


noncomputable instance [Abelian V] : Abelian (Action V G) :=
  abelianOfEquivalence abelianAux.functor


instance mapAction_preadditive [F.Additive] : (F.mapAction G).Additive where


instance mapAction_linear [F.Additive] [F.Linear R] : (F.mapAction G).Linear R where


