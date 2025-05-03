/-- If `relations : Relations A` involved no relation, then it has an obvious
solution in the module `relations.G →₀ A`. -/
@[simps]
noncomputable def solutionFinsupp : relations.Solution (relations.G →₀ A) where
  var g := Finsupp.single g 1
                                         /-
                                           A : Type u
                                           inst✝³ : Ring A
                                           relations : Module.Relations A
                                           M : Type v
                                           inst✝² : AddCommGroup M
                                           inst✝¹ : Module A M
                                           inst✝ : IsEmpty relations.R
                                           r : relations.R
                                           ⊢ Eq ((Finsupp.linearCombination A fun g => Finsupp.single g 1) (relations.rel …
                                         -/
  linearCombination_var_relation r := by exfalso; exact IsEmpty.false r
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- If `relations : Relations A` involves no relations (`[IsEmpty relations.R]`),
then the free module `relations.G →₀ A` satisfies the universal property of the
corresponding module defined by generators (and relations). -/
noncomputable def solutionFinsupp.isPresentationCore :
    Solution.IsPresentationCore.{w} relations.solutionFinsupp where
  desc s := Finsupp.linearCombination _ s.var
                      /-
                        A : Type u
                        inst✝³ : Ring A
                        relations : Module.Relations A
                        M : Type v
                        inst✝² : AddCommGroup M
                        inst✝¹ : Module A M
                        inst✝ : IsEmpty relations.R
                        ⊢ ∀ {N : Type w} [inst : AddCommGroup N] [inst_1 : Module A N] (s : relations. …
                      -/
  postcomp_desc := by aesop
                      /-
                        🎉 no goals
                      -/
                             /-
                               A : Type u
                               inst✝⁵ : Ring A
                               relations : Module.Relations A
                               M : Type v
                               inst✝⁴ : AddCommGroup M
                               inst✝³ : Module A M
                               inst✝² : IsEmpty relations.R
                               N✝ : Type w
                               inst✝¹ : AddCommGroup N✝
                               inst✝ : Module A N✝
                               f✝ f'✝ : LinearMap (RingHom.id A) (Finsupp relations.G A) N✝
                               h : Eq (relations.solutionFinsupp.postcomp f✝) (relations.solutionFinsupp.post …
                               ⊢ Eq f✝ f'✝
                             -/
  postcomp_injective h := by ext; apply Solution.congr_var h
                                  /-
                                    🎉 no goals
                                  -/


lemma solutionFinsupp_isPresentation :
    relations.solutionFinsupp.IsPresentation :=
  (solutionFinsupp.isPresentationCore relations).isPresentation


lemma Solution.IsPresentation.free {solution : relations.Solution M}
    (h : solution.IsPresentation) :
    Module.Free A M :=
  Free.of_equiv ((solutionFinsupp_isPresentation relations).uniq h)


/-- The presentation of the `A`-module `G →₀ A` with generators indexed by `G`,
and no relation. (Note that there is an auxiliary universe parameter `w₁` for the
empty type `R`.) -/
@[simps! G R var]
noncomputable def presentationFinsupp (G : Type w₀) :
    Presentation.{w₀, w₁} A (G →₀ A) where
  G := G
  R := PEmpty.{w₁ + 1}
                 /-
                   A : Type u
                   inst✝² : Ring A
                   relations : Module.Relations A
                   M : Type v
                   inst✝¹ : AddCommGroup M
                   inst✝ : Module A M
                   G : Type w₀
                   ⊢ PEmpty.{w₁ + 1} → Finsupp G A
                 -/
  relation := by rintro ⟨⟩
                 /-
                   🎉 no goals
                 -/
  toSolution := Relations.solutionFinsupp _
  toIsPresentation := Relations.solutionFinsupp_isPresentation _


lemma free_iff_exists_presentation :
    Free A M ↔ ∃ (p : Presentation.{v, w₁} A M), IsEmpty p.R := by
  /-
    A : Type u
    inst✝² : Ring A
    M : Type v
    inst✝¹ : AddCommGroup M
    inst✝ : Module A M
    ⊢ Iff (Module.Free A M) (Exists fun p => IsEmpty p.R)
  -/
  constructor
    /-
      case mp
      A : Type u
      inst✝² : Ring A
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module A M
      ⊢ Module.Free A M → Exists fun p => IsEmpty p.R
    -/
  · rw [free_def.{_, _, v}]
    /-
      case mp
      A : Type u
      inst✝² : Ring A
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module A M
      ⊢ (Exists fun I => Nonempty (Basis I A M)) → Exists fun p => IsEmpty p.R
    -/
    rintro ⟨G, ⟨⟨e⟩⟩⟩
    exact ⟨(presentationFinsupp A G).ofLinearEquiv e.symm,
      by dsimp; infer_instance⟩
    /-
      case mpr
      A : Type u
      inst✝² : Ring A
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module A M
      ⊢ (Exists fun p => IsEmpty p.R) → Module.Free A M
    -/
  · rintro ⟨p, h⟩
    /-
      case mpr.intro
      A : Type u
      inst✝² : Ring A
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module A M
      p : Module.Presentation A M
      h : IsEmpty p.R
      ⊢ Module.Free A M
    -/
    exact p.toIsPresentation.free
    /-
      🎉 no goals
    -/


