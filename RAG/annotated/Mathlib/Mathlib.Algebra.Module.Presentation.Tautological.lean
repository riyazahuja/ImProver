/-- The type which parametrizes the tautological relations in an `A`-module `M`. -/
inductive tautological.R
  | add (m₁ m₂ : M)
  | smul (a : A) (m : M)


/-- The system of equations corresponding to the tautological presentation of an `A`-module. -/
@[simps]
noncomputable def tautologicalRelations : Relations A where
  G := M
  R := tautological.R A M
  relation r := match r with
    | .add m₁ m₂ => Finsupp.single m₁ 1 + Finsupp.single m₂ 1 - Finsupp.single (m₁ + m₂) 1
    | .smul a m => a • Finsupp.single m 1 - Finsupp.single (a • m) 1


variable {A M} in
/-- Solutions of `tautologicalRelations A M` in an `A`-module `N` identify to `M →ₗ[A] N`. -/
def tautologicalRelationsSolutionEquiv {N : Type w} [AddCommGroup N] [Module A N] :
    (tautologicalRelations A M).Solution N ≃ (M →ₗ[A] N) where
  toFun s :=
    { toFun := s.var
      map_add' := fun m₁ m₂ ↦ by
        /-
          A : Type u
          inst✝⁴ : Ring A
          M : Type v
          inst✝³ : AddCommGroup M
          inst✝² : Module A M
          N : Type w
          inst✝¹ : AddCommGroup N
          inst✝ : Module A N
          s : (Module.Presentation.tautologicalRelations A M).Solution N
          m₁ m₂ : M
          ⊢ Eq (s.var (HAdd.hAdd m₁ m₂)) (HAdd.hAdd (s.var m₁) (s.var m₂))
        -/
        symm
        /-
          A : Type u
          inst✝⁴ : Ring A
          M : Type v
          inst✝³ : AddCommGroup M
          inst✝² : Module A M
          N : Type w
          inst✝¹ : AddCommGroup N
          inst✝ : Module A N
          s : (Module.Presentation.tautologicalRelations A M).Solution N
          m₁ m₂ : M
          ⊢ Eq (HAdd.hAdd (s.var m₁) (s.var m₂)) (s.var (HAdd.hAdd m₁ m₂))
        -/
        rw [← sub_eq_zero]
        /-
          A : Type u
          inst✝⁴ : Ring A
          M : Type v
          inst✝³ : AddCommGroup M
          inst✝² : Module A M
          N : Type w
          inst✝¹ : AddCommGroup N
          inst✝ : Module A N
          s : (Module.Presentation.tautologicalRelations A M).Solution N
          m₁ m₂ : M
          ⊢ Eq (HSub.hSub (HAdd.hAdd (s.var m₁) (s.var m₂)) (s.var (HAdd.hAdd m₁ m₂))) 0
        -/
        simpa using s.linearCombination_var_relation (.add m₁ m₂)
        /-
          🎉 no goals
        -/
      map_smul' := fun a m ↦ by
        /-
          A : Type u
          inst✝⁴ : Ring A
          M : Type v
          inst✝³ : AddCommGroup M
          inst✝² : Module A M
          N : Type w
          inst✝¹ : AddCommGroup N
          inst✝ : Module A N
          s : (Module.Presentation.tautologicalRelations A M).Solution N
          a : A
          m : M
          ⊢ Eq ({ toFun := s.var, map_add' := ⋯ }.toFun (HSMul.hSMul a m)) (HSMul.hSMul  …
        -/
        symm
        /-
          A : Type u
          inst✝⁴ : Ring A
          M : Type v
          inst✝³ : AddCommGroup M
          inst✝² : Module A M
          N : Type w
          inst✝¹ : AddCommGroup N
          inst✝ : Module A N
          s : (Module.Presentation.tautologicalRelations A M).Solution N
          a : A
          m : M
          ⊢ Eq (HSMul.hSMul ((RingHom.id A) a) ({ toFun := s.var, map_add' := ⋯ }.toFun  …
        -/
        rw [← sub_eq_zero]
        /-
          A : Type u
          inst✝⁴ : Ring A
          M : Type v
          inst✝³ : AddCommGroup M
          inst✝² : Module A M
          N : Type w
          inst✝¹ : AddCommGroup N
          inst✝ : Module A N
          s : (Module.Presentation.tautologicalRelations A M).Solution N
          a : A
          m : M
          ⊢ Eq (HSub.hSub (HSMul.hSMul ((RingHom.id A) a) ({ toFun := s.var, map_add' := …
        -/
        simpa using s.linearCombination_var_relation (.smul a m) }
        /-
          🎉 no goals
        -/
  invFun f :=
    { var := f
                                           /-
                                             A : Type u
                                             inst✝⁴ : Ring A
                                             M : Type v
                                             inst✝³ : AddCommGroup M
                                             inst✝² : Module A M
                                             N : Type w
                                             inst✝¹ : AddCommGroup N
                                             inst✝ : Module A N
                                             f : LinearMap (RingHom.id A) M N
                                             ⊢ ∀ (r : (Module.Presentation.tautologicalRelations A M).R), Eq ((Finsupp.line …
                                           -/
                                                              /-
                                                                🎉 no goals
                                                              -/
      linearCombination_var_relation := by rintro (_ | _) <;> simp }
                                                              /-
                                                                🎉 no goals
                                                              -/
  left_inv _ := rfl
  right_inv _ := rfl


/-- The obvious solution of `tautologicalRelations A M` in the module `M`. -/
@[simps! var]
def tautologicalSolution : (tautologicalRelations A M).Solution M :=
  tautologicalRelationsSolutionEquiv.symm .id


/-- Any `A`-module admits a tautological presentation by generators and relations. -/
def tautologicalSolutionIsPresentationCore :
    Relations.Solution.IsPresentationCore.{w} (tautologicalSolution A M) where
  desc s := tautologicalRelationsSolutionEquiv s
  postcomp_desc _ := rfl
  postcomp_injective h := by
    /-
      A : Type u
      inst✝⁴ : Ring A
      M : Type v
      inst✝³ : AddCommGroup M
      inst✝² : Module A M
      N✝ : Type w
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) M N✝
      h : Eq ((Module.Presentation.tautologicalSolution A M).postcomp f✝) ((Module.P …
      ⊢ Eq f✝ f'✝
    -/
    ext m
    /-
      case h
      A : Type u
      inst✝⁴ : Ring A
      M : Type v
      inst✝³ : AddCommGroup M
      inst✝² : Module A M
      N✝ : Type w
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) M N✝
      h : Eq ((Module.Presentation.tautologicalSolution A M).postcomp f✝) ((Module.P …
      m : M
      ⊢ Eq (f✝ m) (f'✝ m)
    -/
    exact Relations.Solution.congr_var h m
    /-
      🎉 no goals
    -/


lemma tautologicalSolution_isPresentation :
    (tautologicalSolution A M).IsPresentation :=
  (tautologicalSolutionIsPresentationCore A M).isPresentation


/-- The tautological presentation of any `A`-module `M` by generators and relations.
There is a generator `[m]` for any element `m : M`, and there are two types of relations:
* `[m₁] + [m₂] - [m₁ + m₂] = 0`
* `a • [m] - [a • m] = 0`. -/
@[simps!]
noncomputable def tautological : Presentation A M :=
  ofIsPresentation (tautologicalSolution_isPresentation A M)


