/-- A category has finite products if there exists a limit for every diagram
with shape `Discrete J`, where we have `[Finite J]`.

We require this condition only for `J = Fin n` in the definition, then deduce a version for any
`J : Type*` as a corollary of this definition.
-/
class HasFiniteProducts : Prop where
  /-- `C` has finite products -/
  out (n : ℕ) : HasLimitsOfShape (Discrete (Fin n)) C


/-- If `C` has finite limits then it has finite products. -/
instance (priority := 10) hasFiniteProducts_of_hasFiniteLimits [HasFiniteLimits C] :
    HasFiniteProducts C :=
  ⟨fun _ => inferInstance⟩


instance hasLimitsOfShape_discrete [HasFiniteProducts C] (ι : Type w) [Finite ι] :
    HasLimitsOfShape (Discrete ι) C := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    ι : Type w
    inst✝ : Finite ι
    ⊢ CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete ι) C
  -/
  rcases Finite.exists_equiv_fin ι with ⟨n, ⟨e⟩⟩
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    ι : Type w
    inst✝ : Finite ι
    n : Nat
    e : Equiv ι (Fin n)
    ⊢ CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete ι) C
  -/
  haveI : HasLimitsOfShape (Discrete (Fin n)) C := HasFiniteProducts.out n
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    ι : Type w
    inst✝ : Finite ι
    n : Nat
    e : Equiv ι (Fin n)
    this : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Fin n) …
    ⊢ CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete ι) C
  -/
  exact hasLimitsOfShape_of_equivalence (Discrete.equivalence e.symm)
  /-
    🎉 no goals
  -/


/-- If a category has all products then in particular it has finite products.
-/
theorem hasFiniteProducts_of_hasProducts [HasProducts.{w} C] : HasFiniteProducts C :=
  ⟨fun _ => hasLimitsOfShape_of_equivalence (Discrete.equivalence Equiv.ulift.{w})⟩


/-- A category has finite coproducts if there exists a colimit for every diagram
with shape `Discrete J`, where we have `[Fintype J]`.

We require this condition only for `J = Fin n` in the definition, then deduce a version for any
`J : Type*` as a corollary of this definition.
-/
class HasFiniteCoproducts : Prop where
  /-- `C` has all finite coproducts -/
  out (n : ℕ) : HasColimitsOfShape (Discrete (Fin n)) C

-- attribute [class] HasFiniteCoproducts Porting note: this doesn't seem necessary in Lean 4


instance hasColimitsOfShape_discrete [HasFiniteCoproducts C] (ι : Type w) [Finite ι] :
    HasColimitsOfShape (Discrete ι) C := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    ι : Type w
    inst✝ : Finite ι
    ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete ι) C
  -/
  rcases Finite.exists_equiv_fin ι with ⟨n, ⟨e⟩⟩
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    ι : Type w
    inst✝ : Finite ι
    n : Nat
    e : Equiv ι (Fin n)
    ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete ι) C
  -/
  haveI : HasColimitsOfShape (Discrete (Fin n)) C := HasFiniteCoproducts.out n
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    ι : Type w
    inst✝ : Finite ι
    n : Nat
    e : Equiv ι (Fin n)
    this : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Fin  …
    ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete ι) C
  -/
  exact hasColimitsOfShape_of_equivalence (Discrete.equivalence e.symm)
  /-
    🎉 no goals
  -/


/-- If `C` has finite colimits then it has finite coproducts. -/
instance (priority := 10) hasFiniteCoproducts_of_hasFiniteColimits [HasFiniteColimits C] :
    HasFiniteCoproducts C :=
               /-
                 C : Type u
                 inst✝¹ : CategoryTheory.Category.{v, u} C
                 inst✝ : CategoryTheory.Limits.HasFiniteColimits C
                 J : Nat
                 ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Fin J)) C
               -/
  ⟨fun J => by infer_instance⟩
               /-
                 🎉 no goals
               -/


/-- If a category has all coproducts then in particular it has finite coproducts.
-/
theorem hasFiniteCoproducts_of_hasCoproducts [HasCoproducts.{w} C] : HasFiniteCoproducts C :=
  ⟨fun _ => hasColimitsOfShape_of_equivalence (Discrete.equivalence Equiv.ulift.{w})⟩


