/-- `LinearIndependent R v` states the family of vectors `v` is linearly independent over `R`. -/
def LinearIndependent : Prop :=
  Function.Injective (Finsupp.linearCombination R v)


open Lean PrettyPrinter.Delaborator SubExpr in
/-- Delaborator for `LinearIndependent` that suggests pretty printing with type hints
in case the family of vectors is over a `Set`.

Type hints look like `LinearIndependent fun (v : ↑s) => ↑v` or `LinearIndependent (ι := ↑s) f`,
depending on whether the family is a lambda expression or not. -/
@[app_delab LinearIndependent]
def delabLinearIndependent : Delab :=
  whenPPOption getPPNotation <|
  whenNotPPOption getPPAnalysisSkip <|
  withOptionAtCurrPos `pp.analysis.skip true do
    let e ← getExpr
    guard <| e.isAppOfArity ``LinearIndependent 7
    let some _ := (e.getArg! 0).coeTypeSet? | failure
    let optionsPerPos ← if (e.getArg! 3).isLambda then
      withNaryArg 3 do return (← read).optionsPerPos.setBool (← getPos) pp.funBinderTypes.name true
    else
      withNaryArg 0 do return (← read).optionsPerPos.setBool (← getPos) `pp.analysis.namedArg true
    withTheReader Context ({· with optionsPerPos}) delab


theorem linearIndependent_iff_injective_linearCombination :
    LinearIndependent R v ↔ Function.Injective (Finsupp.linearCombination R v) := Iff.rfl


alias ⟨LinearIndependent.injective_linearCombination, _⟩ :=
  linearIndependent_iff_injective_linearCombination


theorem LinearIndependent.ne_zero [Nontrivial R] (i : ι) (hv : LinearIndependent R v) :
    v i ≠ 0 := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    i : ι
    hv : LinearIndependent R v
    ⊢ Ne (v i) 0
  -/
  intro h
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    i : ι
    hv : LinearIndependent R v
    h : Eq (v i) 0
    ⊢ False
  -/
  have := @hv (Finsupp.single i 1 : ι →₀ R) 0 ?_
    /-
      case refine_2
      ι : Type u'
      R : Type u_2
      M : Type u_4
      v : ι → M
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      inst✝ : Nontrivial R
      i : ι
      hv : LinearIndependent R v
      h : Eq (v i) 0
      this : Eq (Finsupp.single i 1) 0
      ⊢ False
    -/
  · simp at this
    /-
      🎉 no goals
    -/
  /-
    case refine_1
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    i : ι
    hv : LinearIndependent R v
    h : Eq (v i) 0
    ⊢ Eq ((Finsupp.linearCombination R v) (Finsupp.single i 1)) ((Finsupp.linearCo …
  -/
  simpa using h
  /-
    🎉 no goals
  -/



theorem linearIndependent_empty_type [IsEmpty ι] : LinearIndependent R v :=
  Function.injective_of_subsingleton _


variable (R M) in
theorem linearIndependent_empty : LinearIndependent R (fun x => x : (∅ : Set M) → M) :=
  linearIndependent_empty_type


/-- A subfamily of a linearly independent family (i.e., a composition with an injective map) is a
linearly independent family. -/
theorem LinearIndependent.comp (h : LinearIndependent R v) (f : ι' → ι) (hf : Injective f) :
    LinearIndependent R (v ∘ f) := by
  /-
    ι : Type u'
    ι' : Type u_1
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    h : LinearIndependent R v
    f : ι' → ι
    hf : Function.Injective f
    ⊢ LinearIndependent R (Function.comp v f)
  -/
  simpa [Function.comp_def] using Function.Injective.comp h (Finsupp.mapDomain_injective hf)
  /-
    🎉 no goals
  -/


/-- A set of linearly independent vectors in a module `M` over a semiring `K` is also linearly
independent over a subring `R` of `K`.
The implementation uses minimal assumptions about the relationship between `R`, `K` and `M`.
The version where `K` is an `R`-algebra is `LinearIndependent.restrict_scalars_algebras`.
 -/
theorem LinearIndependent.restrict_scalars [Semiring K] [SMulWithZero R K] [Module K M]
    [IsScalarTower R K M] (hinj : Function.Injective fun r : R => r • (1 : K))
    (li : LinearIndependent K v) : LinearIndependent R v := by
  /-
    ι : Type u'
    R : Type u_2
    K : Type u_3
    M : Type u_4
    v : ι → M
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Semiring K
    inst✝² : SMulWithZero R K
    inst✝¹ : Module K M
    inst✝ : IsScalarTower R K M
    hinj : Function.Injective fun r => HSMul.hSMul r 1
    li : LinearIndependent K v
    ⊢ LinearIndependent R v
  -/
  intro x y hxy
  /-
    ι : Type u'
    R : Type u_2
    K : Type u_3
    M : Type u_4
    v : ι → M
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Semiring K
    inst✝² : SMulWithZero R K
    inst✝¹ : Module K M
    inst✝ : IsScalarTower R K M
    hinj : Function.Injective fun r => HSMul.hSMul r 1
    li : LinearIndependent K v
    x y : Finsupp ι R
    hxy : Eq ((Finsupp.linearCombination R v) x) ((Finsupp.linearCombination R v) y)
    ⊢ Eq x y
  -/
  let f := fun r : R => r • (1 : K)
  /-
    ι : Type u'
    R : Type u_2
    K : Type u_3
    M : Type u_4
    v : ι → M
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Semiring K
    inst✝² : SMulWithZero R K
    inst✝¹ : Module K M
    inst✝ : IsScalarTower R K M
    hinj : Function.Injective fun r => HSMul.hSMul r 1
    li : LinearIndependent K v
    x y : Finsupp ι R
    hxy : Eq ((Finsupp.linearCombination R v) x) ((Finsupp.linearCombination R v) y)
    f : R → K := fun r => HSMul.hSMul r 1
    ⊢ Eq x y
  -/
  have := @li (x.mapRange f (by simp [f])) (y.mapRange f (by simp [f])) ?_
    /-
      case refine_2
      ι : Type u'
      R : Type u_2
      K : Type u_3
      M : Type u_4
      v : ι → M
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : Semiring K
      inst✝² : SMulWithZero R K
      inst✝¹ : Module K M
      inst✝ : IsScalarTower R K M
      hinj : Function.Injective fun r => HSMul.hSMul r 1
      li : LinearIndependent K v
      x y : Finsupp ι R
      hxy : Eq ((Finsupp.linearCombination R v) x) ((Finsupp.linearCombination R v) y)
      f : R → K := fun r => HSMul.hSMul r 1
      this : Eq (Finsupp.mapRange f ⋯ x) (Finsupp.mapRange f ⋯ y)
      ⊢ Eq x y
    -/
  · ext i
    /-
      case refine_2.h
      ι : Type u'
      R : Type u_2
      K : Type u_3
      M : Type u_4
      v : ι → M
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : Semiring K
      inst✝² : SMulWithZero R K
      inst✝¹ : Module K M
      inst✝ : IsScalarTower R K M
      hinj : Function.Injective fun r => HSMul.hSMul r 1
      li : LinearIndependent K v
      x y : Finsupp ι R
      hxy : Eq ((Finsupp.linearCombination R v) x) ((Finsupp.linearCombination R v) y)
      f : R → K := fun r => HSMul.hSMul r 1
      this : Eq (Finsupp.mapRange f ⋯ x) (Finsupp.mapRange f ⋯ y)
      i : ι
      ⊢ Eq (x i) (y i)
    -/
    exact hinj congr($this i)
    /-
      🎉 no goals
    -/
  /-
    case refine_1
    ι : Type u'
    R : Type u_2
    K : Type u_3
    M : Type u_4
    v : ι → M
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : Semiring K
    inst✝² : SMulWithZero R K
    inst✝¹ : Module K M
    inst✝ : IsScalarTower R K M
    hinj : Function.Injective fun r => HSMul.hSMul r 1
    li : LinearIndependent K v
    x y : Finsupp ι R
    hxy : Eq ((Finsupp.linearCombination R v) x) ((Finsupp.linearCombination R v) y)
    f : R → K := fun r => HSMul.hSMul r 1
    ⊢ Eq ((Finsupp.linearCombination K v) (Finsupp.mapRange f ⋯ x)) ((Finsupp.line …
  -/
  simpa [Finsupp.linearCombination, f, Finsupp.sum_mapRange_index]
  /-
    🎉 no goals
  -/


theorem linearIndependent_iff_ker :
    LinearIndependent R v ↔ LinearMap.ker (Finsupp.linearCombination R v) = ⊥ :=
  LinearMap.ker_eq_bot.symm


theorem linearIndependent_iff :
    LinearIndependent R v ↔ ∀ l, Finsupp.linearCombination R v l = 0 → l = 0 := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (LinearIndependent R v) (∀ (l : Finsupp ι R), Eq ((Finsupp.linearCombina …
  -/
  simp [linearIndependent_iff_ker, LinearMap.ker_eq_bot']
  /-
    🎉 no goals
  -/


theorem linearIndependent_iff' :
    LinearIndependent R v ↔
      ∀ s : Finset ι, ∀ g : ι → R, ∑ i ∈ s, g i • v i = 0 → ∀ i ∈ s, g i = 0 :=
  linearIndependent_iff.trans
    ⟨fun hf s g hg i his =>
      have h :=
        hf (∑ i ∈ s, Finsupp.single i (g i)) <| by
          /-
            ι : Type u'
            R : Type u_2
            M : Type u_4
            v : ι → M
            inst✝² : Ring R
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            hf : ∀ (l : Finsupp ι R), Eq ((Finsupp.linearCombination R v) l) 0 → Eq l 0
            s : Finset ι
            g : ι → R
            hg : Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) 0
            i : ι
            his : Membership.mem s i
            ⊢ Eq ((Finsupp.linearCombination R v) (s.sum fun i => Finsupp.single i (g i))) 0
          -/
          simpa only [map_sum, Finsupp.linearCombination_single] using hg
          /-
            🎉 no goals
          -/
      calc
        g i = (Finsupp.lapply i : (ι →₀ R) →ₗ[R] R) (Finsupp.single i (g i)) := by
          /-
            ι : Type u'
            R : Type u_2
            M : Type u_4
            v : ι → M
            inst✝² : Ring R
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            hf : ∀ (l : Finsupp ι R), Eq ((Finsupp.linearCombination R v) l) 0 → Eq l 0
            s : Finset ι
            g : ι → R
            hg : Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) 0
            i : ι
            his : Membership.mem s i
            h : Eq (s.sum fun i => Finsupp.single i (g i)) 0
            ⊢ Eq (g i) ((Finsupp.lapply i) (Finsupp.single i (g i)))
          -/
          { rw [Finsupp.lapply_apply, Finsupp.single_eq_same] }
          /-
            🎉 no goals
          -/
        _ = ∑ j ∈ s, (Finsupp.lapply i : (ι →₀ R) →ₗ[R] R) (Finsupp.single j (g j)) :=
          Eq.symm <|
            Finset.sum_eq_single i
                                    /-
                                      ι : Type u'
                                      R : Type u_2
                                      M : Type u_4
                                      v : ι → M
                                      inst✝² : Ring R
                                      inst✝¹ : AddCommGroup M
                                      inst✝ : Module R M
                                      hf : ∀ (l : Finsupp ι R), Eq ((Finsupp.linearCombination R v) l) 0 → Eq l 0
                                      s : Finset ι
                                      g : ι → R
                                      hg : Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) 0
                                      i : ι
                                      his : Membership.mem s i
                                      h : Eq (s.sum fun i => Finsupp.single i (g i)) 0
                                      j : ι
                                      _hjs : Membership.mem s j
                                      hji : Ne j i
                                      ⊢ Eq ((Finsupp.lapply i) (Finsupp.single j (g j))) 0
                                    -/
              (fun j _hjs hji => by rw [Finsupp.lapply_apply, Finsupp.single_eq_of_ne hji])
                                    /-
                                      🎉 no goals
                                    -/
              fun hnis => hnis.elim his
        _ = (∑ j ∈ s, Finsupp.single j (g j)) i := (map_sum ..).symm
        _ = 0 := DFunLike.ext_iff.1 h i,
      fun hf _ hl =>
      Finsupp.ext fun _ =>
        _root_.by_contradiction fun hni => hni <| hf _ _ hl _ <| Finsupp.mem_support_iff.2 hni⟩


theorem linearIndependent_iff'' :
    LinearIndependent R v ↔
      ∀ (s : Finset ι) (g : ι → R), (∀ i ∉ s, g i = 0) →
        ∑ i ∈ s, g i • v i = 0 → ∀ i, g i = 0 := by
  classical
  exact linearIndependent_iff'.trans
    ⟨fun H s g hg hv i => if his : i ∈ s then H s g hv i his else hg i his, fun H s g hg i hi => by
      convert
        H s (fun j => if j ∈ s then g j else 0) (fun j hj => if_neg hj)
          (by simp_rw [ite_smul, zero_smul, Finset.sum_extend_by_zero, hg]) i
      exact (if_pos hi).symm⟩


theorem not_linearIndependent_iff :
    ¬LinearIndependent R v ↔
      ∃ s : Finset ι, ∃ g : ι → R, ∑ i ∈ s, g i • v i = 0 ∧ ∃ i ∈ s, g i ≠ 0 := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Not (LinearIndependent R v)) (Exists fun s => Exists fun g => And (Eq ( …
  -/
  rw [linearIndependent_iff']
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Not (∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) …
  -/
  simp only [exists_prop, not_forall]
  /-
    🎉 no goals
  -/


theorem Fintype.linearIndependent_iff [Fintype ι] :
    LinearIndependent R v ↔ ∀ g : ι → R, ∑ i, g i • v i = 0 → ∀ i, g i = 0 := by
  refine
    ⟨fun H g => by simpa using linearIndependent_iff'.1 H Finset.univ g, fun H =>
      linearIndependent_iff''.2 fun s g hg hs i => H _ ?_ _⟩
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Fintype ι
    H : ∀ (g : ι → R), Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (v i)) 0 → ∀ …
    s : Finset ι
    g : ι → R
    hg : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
    hs : Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) 0
    i : ι
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (v i)) 0
  -/
  rw [← hs]
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Fintype ι
    H : ∀ (g : ι → R), Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (v i)) 0 → ∀ …
    s : Finset ι
    g : ι → R
    hg : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
    hs : Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) 0
    i : ι
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (v i)) (s.sum fun i => HSMul. …
  -/
  refine (Finset.sum_subset (Finset.subset_univ _) fun i _ hi => ?_).symm
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Fintype ι
    H : ∀ (g : ι → R), Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (v i)) 0 → ∀ …
    s : Finset ι
    g : ι → R
    hg : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
    hs : Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) 0
    i✝ i : ι
    x✝ : Membership.mem Finset.univ i
    hi : Not (Membership.mem s i)
    ⊢ Eq (HSMul.hSMul (g i) (v i)) 0
  -/
  rw [hg i hi, zero_smul]
  /-
    🎉 no goals
  -/


/-- A finite family of vectors `v i` is linear independent iff the linear map that sends
`c : ι → R` to `∑ i, c i • v i` has the trivial kernel. -/
theorem Fintype.linearIndependent_iff' [Fintype ι] [DecidableEq ι] :
    LinearIndependent R v ↔
      LinearMap.ker (LinearMap.lsum R (fun _ ↦ R) ℕ fun i ↦ LinearMap.id.smulRight (v i)) = ⊥ := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Iff (LinearIndependent R v) (Eq (LinearMap.ker ((LinearMap.lsum R (fun x =>  …
  -/
  simp [Fintype.linearIndependent_iff, LinearMap.ker_eq_bot', funext_iff]
  /-
    🎉 no goals
  -/


theorem Fintype.not_linearIndependent_iff [Fintype ι] :
    ¬LinearIndependent R v ↔ ∃ g : ι → R, ∑ i, g i • v i = 0 ∧ ∃ i, g i ≠ 0 := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Fintype ι
    ⊢ Iff (Not (LinearIndependent R v)) (Exists fun g => And (Eq (Finset.univ.sum  …
  -/
  simpa using not_iff_not.2 Fintype.linearIndependent_iff
  /-
    🎉 no goals
  -/


lemma LinearIndependent.eq_zero_of_pair {x y : M} (h : LinearIndependent R ![x, y])
    {s t : R} (h' : s • x + t • y = 0) : s = 0 ∧ t = 0 := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    s t : R
    h' : Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0
    ⊢ And (Eq s 0) (Eq t 0)
  -/
  replace h := @h (.single 0 s + .single 1 t) 0 ?_
    /-
      case refine_2
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x y : M
      s t : R
      h' : Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0
      h : Eq (HAdd.hAdd (Finsupp.single 0 s) (Finsupp.single 1 t)) 0
      ⊢ And (Eq s 0) (Eq t 0)
    -/
  · exact ⟨by simpa using congr($h 0), by simpa using congr($h 1)⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_1
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    s t : R
    h' : Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0
    ⊢ Eq ((Finsupp.linearCombination R (Matrix.vecCons x (Matrix.vecCons y Matrix. …
  -/
  simpa
  /-
    🎉 no goals
  -/


/-- Also see `LinearIndependent.pair_iff'` for a simpler version over fields. -/
lemma LinearIndependent.pair_iff {x y : M} :
    LinearIndependent R ![x, y] ↔ ∀ (s t : R), s • x + t • y = 0 → s = 0 ∧ t = 0 := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    ⊢ Iff (LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty …
  -/
  refine ⟨fun h s t hst ↦ h.eq_zero_of_pair hst, fun h ↦ ?_⟩
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : ∀ (s t : R), Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0 → And (E …
    ⊢ LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
  -/
  apply Fintype.linearIndependent_iff.2
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : ∀ (s t : R), Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0 → And (E …
    ⊢ ∀ (g : Fin (Nat.succ 0).succ → R), Eq (Finset.univ.sum fun i => HSMul.hSMul  …
  -/
  intro g hg
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : ∀ (s t : R), Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0 → And (E …
    g : Fin (Nat.succ 0).succ → R
    hg : Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Matrix.vecCons x (Matrix. …
    ⊢ ∀ (i : Fin (Nat.succ 0).succ), Eq (g i) 0
  -/
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons] at hg
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : ∀ (s t : R), Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0 → And (E …
    g : Fin (Nat.succ 0).succ → R
    hg : Eq (HAdd.hAdd (HSMul.hSMul (g 0) x) (HSMul.hSMul (g 1) y)) 0
    ⊢ ∀ (i : Fin (Nat.succ 0).succ), Eq (g i) 0
  -/
  intro i
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : ∀ (s t : R), Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0 → And (E …
    g : Fin (Nat.succ 0).succ → R
    hg : Eq (HAdd.hAdd (HSMul.hSMul (g 0) x) (HSMul.hSMul (g 1) y)) 0
    i : Fin (Nat.succ 0).succ
    ⊢ Eq (g i) 0
  -/
  fin_cases i
  /-
    case «0»
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : ∀ (s t : R), Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0 → And (E …
    g : Fin (Nat.succ 0).succ → R
    hg : Eq (HAdd.hAdd (HSMul.hSMul (g 0) x) (HSMul.hSMul (g 1) y)) 0
    ⊢ Eq (g ((fun i => i) ⟨0, ⋯⟩)) 0
  -/
  exacts [(h _ _ hg).1, (h _ _ hg).2]
  /-
    🎉 no goals
  -/


/-- A family is linearly independent if and only if all of its finite subfamily is
linearly independent. -/
theorem linearIndependent_iff_finset_linearIndependent :
    LinearIndependent R v ↔ ∀ (s : Finset ι), LinearIndependent R (v ∘ (Subtype.val : s → ι)) :=
  ⟨fun H _ ↦ H.comp _ Subtype.val_injective, fun H ↦ linearIndependent_iff'.2 fun s g hg i hi ↦
    Fintype.linearIndependent_iff.1 (H s) (g ∘ Subtype.val)
      (hg ▸ Finset.sum_attach s fun j ↦ g j • v j) ⟨i, hi⟩⟩


theorem LinearIndependent.coe_range (i : LinearIndependent R v) :
                                                  /-
                                                    ι : Type u'
                                                    R : Type u_2
                                                    M : Type u_4
                                                    v : ι → M
                                                    inst✝² : Ring R
                                                    inst✝¹ : AddCommGroup M
                                                    inst✝ : Module R M
                                                    i : LinearIndependent R v
                                                    ⊢ LinearIndependent R Subtype.val
                                                  -/
    LinearIndependent R ((↑) : range v → M) := by simpa using i.comp _ (rangeSplitting_injective v)
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- If `v` is a linearly independent family of vectors and the kernel of a linear map `f` is
disjoint with the submodule spanned by the vectors of `v`, then `f ∘ v` is a linearly independent
family of vectors. See also `LinearIndependent.map'` for a special case assuming `ker f = ⊥`. -/
theorem LinearIndependent.map (hv : LinearIndependent R v) {f : M →ₗ[R] M'}
    (hf_inj : Disjoint (span R (range v)) (LinearMap.ker f)) : LinearIndependent R (f ∘ v) := by
  rw [disjoint_iff_inf_le, ← Set.image_univ, Finsupp.span_image_eq_map_linearCombination,
    map_inf_eq_map_inf_comap, map_le_iff_le_comap, comap_bot, Finsupp.supported_univ, top_inf_eq]
      at hf_inj
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    M' : Type u_5
    v : ι → M
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M
    inst✝ : Module R M'
    hv : LinearIndependent R v
    f : LinearMap (RingHom.id R) M M'
    hf_inj : LE.le (Submodule.comap (Finsupp.linearCombination R v) (LinearMap.ker …
    ⊢ LinearIndependent R (Function.comp (⇑f) v)
  -/
  rw [linearIndependent_iff_ker] at hv ⊢
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    M' : Type u_5
    v : ι → M
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M
    inst✝ : Module R M'
    hv : Eq (LinearMap.ker (Finsupp.linearCombination R v)) Bot.bot
    f : LinearMap (RingHom.id R) M M'
    hf_inj : LE.le (Submodule.comap (Finsupp.linearCombination R v) (LinearMap.ker …
    ⊢ Eq (LinearMap.ker (Finsupp.linearCombination R (Function.comp (⇑f) v))) Bot. …
  -/
  rw [hv, le_bot_iff] at hf_inj
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    M' : Type u_5
    v : ι → M
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M
    inst✝ : Module R M'
    hv : Eq (LinearMap.ker (Finsupp.linearCombination R v)) Bot.bot
    f : LinearMap (RingHom.id R) M M'
    hf_inj : Eq (Submodule.comap (Finsupp.linearCombination R v) (LinearMap.ker f) …
    ⊢ Eq (LinearMap.ker (Finsupp.linearCombination R (Function.comp (⇑f) v))) Bot. …
  -/
  rw [Finsupp.linearCombination_linear_comp, LinearMap.ker_comp, hf_inj]
  /-
    🎉 no goals
  -/


/-- If `v` is an injective family of vectors such that `f ∘ v` is linearly independent, then `v`
    spans a submodule disjoint from the kernel of `f` -/
theorem Submodule.range_ker_disjoint {f : M →ₗ[R] M'}
    (hv : LinearIndependent R (f ∘ v)) :
    Disjoint (span R (range v)) (LinearMap.ker f) := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    M' : Type u_5
    v : ι → M
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    hv : LinearIndependent R (Function.comp (⇑f) v)
    ⊢ Disjoint (Submodule.span R (Set.range v)) (LinearMap.ker f)
  -/
  rw [linearIndependent_iff_ker, Finsupp.linearCombination_linear_comp, LinearMap.ker_comp] at hv
  rw [disjoint_iff_inf_le, ← Set.image_univ, Finsupp.span_image_eq_map_linearCombination,
    map_inf_eq_map_inf_comap, hv, inf_bot_eq, map_bot]


/-- An injective linear map sends linearly independent families of vectors to linearly independent
families of vectors. See also `LinearIndependent.map` for a more general statement. -/
theorem LinearIndependent.map' (hv : LinearIndependent R v) (f : M →ₗ[R] M')
    (hf_inj : LinearMap.ker f = ⊥) : LinearIndependent R (f ∘ v) :=
               /-
                 ι : Type u'
                 R : Type u_2
                 M : Type u_4
                 M' : Type u_5
                 v : ι → M
                 inst✝⁴ : Ring R
                 inst✝³ : AddCommGroup M
                 inst✝² : AddCommGroup M'
                 inst✝¹ : Module R M
                 inst✝ : Module R M'
                 hv : LinearIndependent R v
                 f : LinearMap (RingHom.id R) M M'
                 hf_inj : Eq (LinearMap.ker f) Bot.bot
                 ⊢ Disjoint (Submodule.span R (Set.range v)) (LinearMap.ker f)
               -/
  hv.map <| by simp [hf_inj]
               /-
                 🎉 no goals
               -/


/-- If `M / R` and `M' / R'` are modules, `i : R' → R` is a map, `j : M →+ M'` is a monoid map,
such that they send non-zero elements to non-zero elements, and compatible with the scalar
multiplications on `M` and `M'`, then `j` sends linearly independent families of vectors to
linearly independent families of vectors. As a special case, taking `R = R'`
it is `LinearIndependent.map'`. -/
theorem LinearIndependent.map_of_injective_injective {R' : Type*} {M' : Type*}
    [Ring R'] [AddCommGroup M'] [Module R' M'] (hv : LinearIndependent R v)
    (i : R' → R) (j : M →+ M') (hi : ∀ r, i r = 0 → r = 0) (hj : ∀ m, j m = 0 → m = 0)
    (hc : ∀ (r : R') (m : M), j (i r • m) = r • j m) : LinearIndependent R' (j ∘ v) := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    R' : Type u_6
    M' : Type u_7
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R' M'
    hv : LinearIndependent R v
    i : R' → R
    j : AddMonoidHom M M'
    hi : ∀ (r : R'), Eq (i r) 0 → Eq r 0
    hj : ∀ (m : M), Eq (j m) 0 → Eq m 0
    hc : ∀ (r : R') (m : M), Eq (j (HSMul.hSMul (i r) m)) (HSMul.hSMul r (j m))
    ⊢ LinearIndependent R' (Function.comp (⇑j) v)
  -/
  rw [linearIndependent_iff'] at hv ⊢
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    R' : Type u_6
    M' : Type u_7
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R' M'
    hv : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    i : R' → R
    j : AddMonoidHom M M'
    hi : ∀ (r : R'), Eq (i r) 0 → Eq r 0
    hj : ∀ (m : M), Eq (j m) 0 → Eq m 0
    hc : ∀ (r : R') (m : M), Eq (j (HSMul.hSMul (i r) m)) (HSMul.hSMul r (j m))
    ⊢ ∀ (s : Finset ι) (g : ι → R'), Eq (s.sum fun i => HSMul.hSMul (g i) (Functio …
  -/
  intro S r' H s hs
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    R' : Type u_6
    M' : Type u_7
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R' M'
    hv : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    i : R' → R
    j : AddMonoidHom M M'
    hi : ∀ (r : R'), Eq (i r) 0 → Eq r 0
    hj : ∀ (m : M), Eq (j m) 0 → Eq m 0
    hc : ∀ (r : R') (m : M), Eq (j (HSMul.hSMul (i r) m)) (HSMul.hSMul r (j m))
    S : Finset ι
    r' : ι → R'
    H : Eq (S.sum fun i => HSMul.hSMul (r' i) (Function.comp (⇑j) v i)) 0
    s : ι
    hs : Membership.mem S s
    ⊢ Eq (r' s) 0
  -/
  simp_rw [comp_apply, ← hc, ← map_sum] at H
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    R' : Type u_6
    M' : Type u_7
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R' M'
    hv : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    i : R' → R
    j : AddMonoidHom M M'
    hi : ∀ (r : R'), Eq (i r) 0 → Eq r 0
    hj : ∀ (m : M), Eq (j m) 0 → Eq m 0
    hc : ∀ (r : R') (m : M), Eq (j (HSMul.hSMul (i r) m)) (HSMul.hSMul r (j m))
    S : Finset ι
    r' : ι → R'
    s : ι
    hs : Membership.mem S s
    H : Eq (j (S.sum fun x => HSMul.hSMul (i (r' x)) (v x))) 0
    ⊢ Eq (r' s) 0
  -/
  exact hi _ <| hv _ _ (hj _ H) s hs
  /-
    🎉 no goals
  -/


/-- If `M / R` and `M' / R'` are modules, `i : R → R'` is a surjective map which maps zero to zero,
`j : M →+ M'` is a monoid map which sends non-zero elements to non-zero elements, such that the
scalar multiplications on `M` and `M'` are compatible, then `j` sends linearly independent families
of vectors to linearly independent families of vectors. As a special case, taking `R = R'`
it is `LinearIndependent.map'`. -/
theorem LinearIndependent.map_of_surjective_injective {R' : Type*} {M' : Type*}
    [Ring R'] [AddCommGroup M'] [Module R' M'] (hv : LinearIndependent R v)
    (i : ZeroHom R R') (j : M →+ M') (hi : Surjective i) (hj : ∀ m, j m = 0 → m = 0)
    (hc : ∀ (r : R) (m : M), j (r • m) = i r • j m) : LinearIndependent R' (j ∘ v) := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    R' : Type u_6
    M' : Type u_7
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R' M'
    hv : LinearIndependent R v
    i : ZeroHom R R'
    j : AddMonoidHom M M'
    hi : Function.Surjective ⇑i
    hj : ∀ (m : M), Eq (j m) 0 → Eq m 0
    hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
    ⊢ LinearIndependent R' (Function.comp (⇑j) v)
  -/
  obtain ⟨i', hi'⟩ := hi.hasRightInverse
  /-
    case intro
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    R' : Type u_6
    M' : Type u_7
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R' M'
    hv : LinearIndependent R v
    i : ZeroHom R R'
    j : AddMonoidHom M M'
    hi : Function.Surjective ⇑i
    hj : ∀ (m : M), Eq (j m) 0 → Eq m 0
    hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
    i' : R' → R
    hi' : Function.RightInverse i' ⇑i
    ⊢ LinearIndependent R' (Function.comp (⇑j) v)
  -/
  refine hv.map_of_injective_injective i' j (fun _ h ↦ ?_) hj fun r m ↦ ?_
    /-
      case intro.refine_1
      ι : Type u'
      R : Type u_2
      M : Type u_4
      v : ι → M
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      R' : Type u_6
      M' : Type u_7
      inst✝² : Ring R'
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R' M'
      hv : LinearIndependent R v
      i : ZeroHom R R'
      j : AddMonoidHom M M'
      hi : Function.Surjective ⇑i
      hj : ∀ (m : M), Eq (j m) 0 → Eq m 0
      hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
      i' : R' → R
      hi' : Function.RightInverse i' ⇑i
      x✝ : R'
      h : Eq (i' x✝) 0
      ⊢ Eq x✝ 0
    -/
  · apply_fun i at h
    /-
      case intro.refine_1
      ι : Type u'
      R : Type u_2
      M : Type u_4
      v : ι → M
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      R' : Type u_6
      M' : Type u_7
      inst✝² : Ring R'
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R' M'
      hv : LinearIndependent R v
      i : ZeroHom R R'
      j : AddMonoidHom M M'
      hi : Function.Surjective ⇑i
      hj : ∀ (m : M), Eq (j m) 0 → Eq m 0
      hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
      i' : R' → R
      hi' : Function.RightInverse i' ⇑i
      x✝ : R'
      h : Eq (i (i' x✝)) (i 0)
      ⊢ Eq x✝ 0
    -/
    rwa [hi', i.map_zero] at h
    /-
      🎉 no goals
    -/
  /-
    case intro.refine_2
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    R' : Type u_6
    M' : Type u_7
    inst✝² : Ring R'
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R' M'
    hv : LinearIndependent R v
    i : ZeroHom R R'
    j : AddMonoidHom M M'
    hi : Function.Surjective ⇑i
    hj : ∀ (m : M), Eq (j m) 0 → Eq m 0
    hc : ∀ (r : R) (m : M), Eq (j (HSMul.hSMul r m)) (HSMul.hSMul (i r) (j m))
    i' : R' → R
    hi' : Function.RightInverse i' ⇑i
    r : R'
    m : M
    ⊢ Eq (j (HSMul.hSMul (i' r) m)) (HSMul.hSMul r (j m))
  -/
  rw [hc (i' r) m, hi']
  /-
    🎉 no goals
  -/


/-- If the image of a family of vectors under a linear map is linearly independent, then so is
the original family. -/
theorem LinearIndependent.of_comp (f : M →ₗ[R] M') (hfv : LinearIndependent R (f ∘ v)) :
    LinearIndependent R v :=
  linearIndependent_iff'.2 fun s g hg i his =>
    have : (∑ i ∈ s, g i • f (v i)) = 0 := by
      /-
        ι : Type u'
        R : Type u_2
        M : Type u_4
        M' : Type u_5
        v : ι → M
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup M'
        inst✝¹ : Module R M
        inst✝ : Module R M'
        f : LinearMap (RingHom.id R) M M'
        hfv : LinearIndependent R (Function.comp (⇑f) v)
        s : Finset ι
        g : ι → R
        hg : Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) 0
        i : ι
        his : Membership.mem s i
        ⊢ Eq (s.sum fun i => HSMul.hSMul (g i) (f (v i))) 0
      -/
      simp_rw [← map_smul, ← map_sum, hg, f.map_zero]
      /-
        🎉 no goals
      -/
    linearIndependent_iff'.1 hfv s g this i his


/-- If `f` is an injective linear map, then the family `f ∘ v` is linearly independent
if and only if the family `v` is linearly independent. -/
protected theorem LinearMap.linearIndependent_iff (f : M →ₗ[R] M') (hf_inj : LinearMap.ker f = ⊥) :
    LinearIndependent R (f ∘ v) ↔ LinearIndependent R v :=
                                              /-
                                                ι : Type u'
                                                R : Type u_2
                                                M : Type u_4
                                                M' : Type u_5
                                                v : ι → M
                                                inst✝⁴ : Ring R
                                                inst✝³ : AddCommGroup M
                                                inst✝² : AddCommGroup M'
                                                inst✝¹ : Module R M
                                                inst✝ : Module R M'
                                                f : LinearMap (RingHom.id R) M M'
                                                hf_inj : Eq (LinearMap.ker f) Bot.bot
                                                h : LinearIndependent R v
                                                ⊢ Disjoint (Submodule.span R (Set.range v)) (LinearMap.ker f)
                                              -/
  ⟨fun h => h.of_comp f, fun h => h.map <| by simp only [hf_inj, disjoint_bot_right]⟩
                                              /-
                                                🎉 no goals
                                              -/


@[nontriviality]
theorem linearIndependent_of_subsingleton [Subsingleton R] : LinearIndependent R v :=
  linearIndependent_iff.2 fun _l _hl => Subsingleton.elim _ _


theorem linearIndependent_equiv (e : ι ≃ ι') {f : ι' → M} :
    LinearIndependent R (f ∘ e) ↔ LinearIndependent R f :=
  ⟨fun h => Function.comp_id f ▸ e.self_comp_symm ▸ h.comp _ e.symm.injective, fun h =>
    h.comp _ e.injective⟩


theorem linearIndependent_equiv' (e : ι ≃ ι') {f : ι' → M} {g : ι → M} (h : f ∘ e = g) :
    LinearIndependent R g ↔ LinearIndependent R f :=
  h ▸ linearIndependent_equiv e


theorem linearIndependent_subtype_range {ι} {f : ι → M} (hf : Injective f) :
    LinearIndependent R ((↑) : range f → M) ↔ LinearIndependent R f :=
  Iff.symm <| linearIndependent_equiv' (Equiv.ofInjective f hf) rfl


alias ⟨LinearIndependent.of_subtype_range, _⟩ := linearIndependent_subtype_range


theorem linearIndependent_image {ι} {s : Set ι} {f : ι → M} (hf : Set.InjOn f s) :
    (LinearIndependent R fun x : s => f x) ↔ LinearIndependent R fun x : f '' s => (x : M) :=
  linearIndependent_equiv' (Equiv.Set.imageOfInjOn _ _ hf) rfl


theorem linearIndependent_span (hs : LinearIndependent R v) :
    LinearIndependent R (M := span R (range v))
      (fun i : ι => ⟨v i, subset_span (mem_range_self i)⟩) :=
  LinearIndependent.of_comp (span R (range v)).subtype hs


/-- See `LinearIndependent.fin_cons` for a family of elements in a vector space. -/
theorem LinearIndependent.fin_cons' {m : ℕ} (x : M) (v : Fin m → M) (hli : LinearIndependent R v)
    (x_ortho : ∀ (c : R) (y : Submodule.span R (Set.range v)), c • x + y = (0 : M) → c = 0) :
    LinearIndependent R (Fin.cons x v : Fin m.succ → M) := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : Nat
    x : M
    v : Fin m → M
    hli : LinearIndependent R v
    x_ortho : ∀ (c : R) (y : Subtype fun x => Membership.mem (Submodule.span R (Se …
    ⊢ LinearIndependent R (Fin.cons x v)
  -/
  rw [Fintype.linearIndependent_iff] at hli ⊢
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : Nat
    x : M
    v : Fin m → M
    hli : ∀ (g : Fin m → R), Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (v i)) …
    x_ortho : ∀ (c : R) (y : Subtype fun x => Membership.mem (Submodule.span R (Se …
    ⊢ ∀ (g : Fin (HAdd.hAdd m 1) → R), Eq (Finset.univ.sum fun i => HSMul.hSMul (g …
  -/
  rintro g total_eq j
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : Nat
    x : M
    v : Fin m → M
    hli : ∀ (g : Fin m → R), Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (v i)) …
    x_ortho : ∀ (c : R) (y : Subtype fun x => Membership.mem (Submodule.span R (Se …
    g : Fin (HAdd.hAdd m 1) → R
    total_eq : Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Fin.cons x v i)) 0
    j : Fin (HAdd.hAdd m 1)
    ⊢ Eq (g j) 0
  -/
  simp_rw [Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ] at total_eq
  have : g 0 = 0 := by
    refine x_ortho (g 0) ⟨∑ i : Fin m, g i.succ • v i, ?_⟩ total_eq
    exact sum_mem fun i _ => smul_mem _ _ (subset_span ⟨i, rfl⟩)
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : Nat
    x : M
    v : Fin m → M
    hli : ∀ (g : Fin m → R), Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (v i)) …
    x_ortho : ∀ (c : R) (y : Subtype fun x => Membership.mem (Submodule.span R (Se …
    g : Fin (HAdd.hAdd m 1) → R
    j : Fin (HAdd.hAdd m 1)
    total_eq : Eq (HAdd.hAdd (HSMul.hSMul (g 0) x) (Finset.univ.sum fun x => HSMul …
    this : Eq (g 0) 0
    ⊢ Eq (g j) 0
  -/
  rw [this, zero_smul, zero_add] at total_eq
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : Nat
    x : M
    v : Fin m → M
    hli : ∀ (g : Fin m → R), Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (v i)) …
    x_ortho : ∀ (c : R) (y : Subtype fun x => Membership.mem (Submodule.span R (Se …
    g : Fin (HAdd.hAdd m 1) → R
    j : Fin (HAdd.hAdd m 1)
    total_eq : Eq (Finset.univ.sum fun x => HSMul.hSMul (g x.succ) (v x)) 0
    this : Eq (g 0) 0
    ⊢ Eq (g j) 0
  -/
  exact Fin.cases this (hli _ total_eq) j
  /-
    🎉 no goals
  -/


/-- Every finite subset of a linearly independent set is linearly independent. -/
theorem linearIndependent_finset_map_embedding_subtype (s : Set M)
    (li : LinearIndependent R ((↑) : s → M)) (t : Finset s) :
    LinearIndependent R ((↑) : Finset.map (Embedding.subtype s) t → M) := by
  let f : t.map (Embedding.subtype s) → s := fun x =>
    ⟨x.1, by
      obtain ⟨x, h⟩ := x
      rw [Finset.mem_map] at h
      obtain ⟨a, _ha, rfl⟩ := h
      simp only [Subtype.coe_prop, Embedding.coe_subtype]⟩
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set M
    li : LinearIndependent R Subtype.val
    t : Finset ↑s
    f : (Subtype fun x => Membership.mem (Finset.map (Function.Embedding.subtype s …
    ⊢ LinearIndependent R Subtype.val
  -/
  convert LinearIndependent.comp li f ?_
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set M
    li : LinearIndependent R Subtype.val
    t : Finset ↑s
    f : (Subtype fun x => Membership.mem (Finset.map (Function.Embedding.subtype s …
    ⊢ Function.Injective f
  -/
  rintro ⟨x, hx⟩ ⟨y, hy⟩
  /-
    case mk.mk
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set M
    li : LinearIndependent R Subtype.val
    t : Finset ↑s
    f : (Subtype fun x => Membership.mem (Finset.map (Function.Embedding.subtype s …
    x : M
    hx : Membership.mem (Finset.map (Function.Embedding.subtype s) t) x
    y : M
    hy : Membership.mem (Finset.map (Function.Embedding.subtype s) t) y
    ⊢ Eq (f ⟨x, hx⟩) (f ⟨y, hy⟩) → Eq ⟨x, hx⟩ ⟨y, hy⟩
  -/
  rw [Finset.mem_map] at hx hy
  /-
    case mk.mk
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set M
    li : LinearIndependent R Subtype.val
    t : Finset ↑s
    f : (Subtype fun x => Membership.mem (Finset.map (Function.Embedding.subtype s …
    x : M
    hx✝ : Membership.mem (Finset.map (Function.Embedding.subtype s) t) x
    hx : Exists fun a => And (Membership.mem t a) (Eq ((Function.Embedding.subtype …
    y : M
    hy✝ : Membership.mem (Finset.map (Function.Embedding.subtype s) t) y
    hy : Exists fun a => And (Membership.mem t a) (Eq ((Function.Embedding.subtype …
    ⊢ Eq (f ⟨x, hx✝⟩) (f ⟨y, hy✝⟩) → Eq ⟨x, hx✝⟩ ⟨y, hy✝⟩
  -/
  obtain ⟨a, _ha, rfl⟩ := hx
  /-
    case mk.mk.intro.intro
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set M
    li : LinearIndependent R Subtype.val
    t : Finset ↑s
    f : (Subtype fun x => Membership.mem (Finset.map (Function.Embedding.subtype s …
    y : M
    hy✝ : Membership.mem (Finset.map (Function.Embedding.subtype s) t) y
    hy : Exists fun a => And (Membership.mem t a) (Eq ((Function.Embedding.subtype …
    a : Subtype s
    _ha : Membership.mem t a
    hx : Membership.mem (Finset.map (Function.Embedding.subtype s) t) ((Function.E …
    ⊢ Eq (f ⟨(Function.Embedding.subtype s) a, hx⟩) (f ⟨y, hy✝⟩) → Eq ⟨(Function.E …
  -/
  obtain ⟨b, _hb, rfl⟩ := hy
  /-
    case mk.mk.intro.intro.intro.intro
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set M
    li : LinearIndependent R Subtype.val
    t : Finset ↑s
    f : (Subtype fun x => Membership.mem (Finset.map (Function.Embedding.subtype s …
    a : Subtype s
    _ha : Membership.mem t a
    hx : Membership.mem (Finset.map (Function.Embedding.subtype s) t) ((Function.E …
    b : Subtype s
    _hb : Membership.mem t b
    hy : Membership.mem (Finset.map (Function.Embedding.subtype s) t) ((Function.E …
    ⊢ Eq (f ⟨(Function.Embedding.subtype s) a, hx⟩) (f ⟨(Function.Embedding.subtyp …
  -/
  simp only [f, imp_self, Subtype.mk_eq_mk]
  /-
    🎉 no goals
  -/



theorem linearIndependent_comp_subtype {s : Set ι} :
    LinearIndependent R (v ∘ (↑) : s → M) ↔
      ∀ l ∈ Finsupp.supported R R s, (Finsupp.linearCombination R v) l = 0 → l = 0 := by
  simp only [linearIndependent_iff, (· ∘ ·), Finsupp.mem_supported, Finsupp.linearCombination_apply,
    Set.subset_def, Finset.mem_coe]
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set ι
    ⊢ Iff (∀ (l : Finsupp (↑s) R), Eq (l.sum fun i a => HSMul.hSMul a (v ↑i)) 0 →  …
  -/
  constructor
    /-
      case mp
      ι : Type u'
      R : Type u_2
      M : Type u_4
      v : ι → M
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : Set ι
      ⊢ (∀ (l : Finsupp (↑s) R), Eq (l.sum fun i a => HSMul.hSMul a (v ↑i)) 0 → Eq l …
    -/
  · intro h l hl₁ hl₂
    exact (Finsupp.subtypeDomain_eq_zero_iff hl₁).1 <|
      h (l.subtypeDomain s) ((Finsupp.sum_subtypeDomain_index hl₁).trans hl₂)
    /-
      case mpr
      ι : Type u'
      R : Type u_2
      M : Type u_4
      v : ι → M
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : Set ι
      ⊢ (∀ (l : Finsupp ι R), (∀ (x : ι), Membership.mem l.support x → Membership.me …
    -/
  · intro h l hl
    /-
      case mpr
      ι : Type u'
      R : Type u_2
      M : Type u_4
      v : ι → M
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : Set ι
      h : ∀ (l : Finsupp ι R), (∀ (x : ι), Membership.mem l.support x → Membership.m …
      l : Finsupp (↑s) R
      hl : Eq (l.sum fun i a => HSMul.hSMul a (v ↑i)) 0
      ⊢ Eq l 0
    -/
    refine Finsupp.embDomain_eq_zero.1 (h (l.embDomain <| Function.Embedding.subtype s) ?_ ?_)
      /-
        case mpr.refine_1
        ι : Type u'
        R : Type u_2
        M : Type u_4
        v : ι → M
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        s : Set ι
        h : ∀ (l : Finsupp ι R), (∀ (x : ι), Membership.mem l.support x → Membership.m …
        l : Finsupp (↑s) R
        hl : Eq (l.sum fun i a => HSMul.hSMul a (v ↑i)) 0
        ⊢ ∀ (x : ι), Membership.mem (Finsupp.embDomain (Function.Embedding.subtype s)  …
      -/
    · suffices ∀ i hi, ¬l ⟨i, hi⟩ = 0 → i ∈ s by simpa
      /-
        case mpr.refine_1
        ι : Type u'
        R : Type u_2
        M : Type u_4
        v : ι → M
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        s : Set ι
        h : ∀ (l : Finsupp ι R), (∀ (x : ι), Membership.mem l.support x → Membership.m …
        l : Finsupp (↑s) R
        hl : Eq (l.sum fun i a => HSMul.hSMul a (v ↑i)) 0
        ⊢ ∀ (i : ι) (hi : Membership.mem s i), Not (Eq (l ⟨i, hi⟩) 0) → Membership.mem …
      -/
      intros
      /-
        case mpr.refine_1
        ι : Type u'
        R : Type u_2
        M : Type u_4
        v : ι → M
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        s : Set ι
        h : ∀ (l : Finsupp ι R), (∀ (x : ι), Membership.mem l.support x → Membership.m …
        l : Finsupp (↑s) R
        hl : Eq (l.sum fun i a => HSMul.hSMul a (v ↑i)) 0
        i✝ : ι
        hi✝ : Membership.mem s i✝
        a✝ : Not (Eq (l ⟨i✝, hi✝⟩) 0)
        ⊢ Membership.mem s i✝
      -/
      assumption
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_2
        ι : Type u'
        R : Type u_2
        M : Type u_4
        v : ι → M
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        s : Set ι
        h : ∀ (l : Finsupp ι R), (∀ (x : ι), Membership.mem l.support x → Membership.m …
        l : Finsupp (↑s) R
        hl : Eq (l.sum fun i a => HSMul.hSMul a (v ↑i)) 0
        ⊢ Eq ((Finsupp.embDomain (Function.Embedding.subtype s) l).sum fun i a => HSMu …
      -/
    · rwa [Finsupp.embDomain_eq_mapDomain, Finsupp.sum_mapDomain_index]
      /-
        case mpr.refine_2.h_zero
        ι : Type u'
        R : Type u_2
        M : Type u_4
        v : ι → M
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        s : Set ι
        h : ∀ (l : Finsupp ι R), (∀ (x : ι), Membership.mem l.support x → Membership.m …
        l : Finsupp (↑s) R
        hl : Eq (l.sum fun i a => HSMul.hSMul a (v ↑i)) 0
        ⊢ ∀ (b : ι), Eq (HSMul.hSMul 0 (v b)) 0
      -/
      exacts [fun _ => zero_smul _ _, fun _ _ _ => add_smul _ _ _]
      /-
        🎉 no goals
      -/


theorem linearDependent_comp_subtype' {s : Set ι} :
    ¬LinearIndependent R (v ∘ (↑) : s → M) ↔
      ∃ f : ι →₀ R, f ∈ Finsupp.supported R R s ∧ Finsupp.linearCombination R v f = 0 ∧ f ≠ 0 := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set ι
    ⊢ Iff (Not (LinearIndependent R (Function.comp v Subtype.val))) (Exists fun f  …
  -/
  simp [linearIndependent_comp_subtype, and_left_comm]
  /-
    🎉 no goals
  -/


/-- A version of `linearDependent_comp_subtype'` with `Finsupp.linearCombination` unfolded. -/
theorem linearDependent_comp_subtype {s : Set ι} :
    ¬LinearIndependent R (v ∘ (↑) : s → M) ↔
      ∃ f : ι →₀ R, f ∈ Finsupp.supported R R s ∧ ∑ i ∈ f.support, f i • v i = 0 ∧ f ≠ 0 :=
  linearDependent_comp_subtype'


theorem linearIndependent_subtype {s : Set M} :
    LinearIndependent R (fun x => x : s → M) ↔
      ∀ l ∈ Finsupp.supported R R s, (Finsupp.linearCombination R id) l = 0 → l = 0 := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set M
    ⊢ Iff (LinearIndependent R fun x => ↑x) (∀ (l : Finsupp M R), Membership.mem ( …
  -/
  apply linearIndependent_comp_subtype (v := id)
  /-
    🎉 no goals
  -/


theorem linearIndependent_comp_subtype_disjoint {s : Set ι} :
    LinearIndependent R (v ∘ (↑) : s → M) ↔
      Disjoint (Finsupp.supported R R s) (LinearMap.ker <| Finsupp.linearCombination R v) := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set ι
    ⊢ Iff (LinearIndependent R (Function.comp v Subtype.val)) (Disjoint (Finsupp.s …
  -/
  rw [linearIndependent_comp_subtype, LinearMap.disjoint_ker]
  /-
    🎉 no goals
  -/


theorem linearIndependent_subtype_disjoint {s : Set M} :
    LinearIndependent R (fun x => x : s → M) ↔
      Disjoint (Finsupp.supported R R s) (LinearMap.ker <| Finsupp.linearCombination R id) := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set M
    ⊢ Iff (LinearIndependent R fun x => ↑x) (Disjoint (Finsupp.supported R R s) (L …
  -/
  apply linearIndependent_comp_subtype_disjoint (v := id)
  /-
    🎉 no goals
  -/


theorem linearIndependent_iff_linearCombinationOn {s : Set M} :
    LinearIndependent R (fun x => x : s → M) ↔
    (LinearMap.ker <| Finsupp.linearCombinationOn M M R id s) = ⊥ := by
  rw [Finsupp.linearCombinationOn, LinearMap.ker, LinearMap.comap_codRestrict, Submodule.map_bot,
      comap_bot, LinearMap.ker_comp, linearIndependent_subtype_disjoint, disjoint_iff_inf_le,
      ← map_comap_subtype, map_le_iff_le_comap, comap_bot, ker_subtype, le_bot_iff]


@[deprecated (since := "2024-08-29")] alias linearIndependent_iff_totalOn :=
  linearIndependent_iff_linearCombinationOn


theorem LinearIndependent.restrict_of_comp_subtype {s : Set ι}
    (hs : LinearIndependent R (v ∘ (↑) : s → M)) : LinearIndependent R (s.restrict v) :=
  hs


theorem LinearIndependent.mono {t s : Set M} (h : t ⊆ s) :
    LinearIndependent R (fun x => x : s → M) → LinearIndependent R (fun x => x : t → M) := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    t s : Set M
    h : HasSubset.Subset t s
    ⊢ (LinearIndependent R fun x => ↑x) → LinearIndependent R fun x => ↑x
  -/
  simp only [linearIndependent_subtype_disjoint]
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    t s : Set M
    h : HasSubset.Subset t s
    ⊢ Disjoint (Finsupp.supported R R s) (LinearMap.ker (Finsupp.linearCombination …
  -/
  exact Disjoint.mono_left (Finsupp.supported_mono h)
  /-
    🎉 no goals
  -/


theorem linearIndependent_of_finite (s : Set M)
    (H : ∀ t ⊆ s, Set.Finite t → LinearIndependent R (fun x => x : t → M)) :
    LinearIndependent R (fun x => x : s → M) :=
  linearIndependent_subtype.2 fun l hl =>
    linearIndependent_subtype.1 (H _ hl (Finset.finite_toSet _)) l (Subset.refl _)


theorem linearIndependent_iUnion_of_directed {η : Type*} {s : η → Set M} (hs : Directed (· ⊆ ·) s)
    (h : ∀ i, LinearIndependent R (fun x => x : s i → M)) :
    LinearIndependent R (fun x => x : (⋃ i, s i) → M) := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    η : Type u_6
    s : η → Set M
    hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (i : η), LinearIndependent R fun x => ↑x
    ⊢ LinearIndependent R fun x => ↑x
  -/
  by_cases hη : Nonempty η
    /-
      case pos
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      η : Type u_6
      s : η → Set M
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), LinearIndependent R fun x => ↑x
      hη : Nonempty η
      ⊢ LinearIndependent R fun x => ↑x
    -/
  · refine linearIndependent_of_finite (⋃ i, s i) fun t ht ft => ?_
    /-
      case pos
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      η : Type u_6
      s : η → Set M
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), LinearIndependent R fun x => ↑x
      hη : Nonempty η
      t : Set M
      ht : HasSubset.Subset t (Set.iUnion fun i => s i)
      ft : t.Finite
      ⊢ LinearIndependent R fun x => ↑x
    -/
    rcases finite_subset_iUnion ft ht with ⟨I, fi, hI⟩
    /-
      case pos.intro.intro
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      η : Type u_6
      s : η → Set M
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), LinearIndependent R fun x => ↑x
      hη : Nonempty η
      t : Set M
      ht : HasSubset.Subset t (Set.iUnion fun i => s i)
      ft : t.Finite
      I : Set η
      fi : I.Finite
      hI : HasSubset.Subset t (Set.iUnion fun i => Set.iUnion fun h => s i)
      ⊢ LinearIndependent R fun x => ↑x
    -/
    rcases hs.finset_le fi.toFinset with ⟨i, hi⟩
    /-
      case pos.intro.intro.intro
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      η : Type u_6
      s : η → Set M
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), LinearIndependent R fun x => ↑x
      hη : Nonempty η
      t : Set M
      ht : HasSubset.Subset t (Set.iUnion fun i => s i)
      ft : t.Finite
      I : Set η
      fi : I.Finite
      hI : HasSubset.Subset t (Set.iUnion fun i => Set.iUnion fun h => s i)
      i : η
      hi : ∀ (i_1 : η), Membership.mem fi.toFinset i_1 → HasSubset.Subset (s i_1) (s …
      ⊢ LinearIndependent R fun x => ↑x
    -/
    exact (h i).mono (Subset.trans hI <| iUnion₂_subset fun j hj => hi j (fi.mem_toFinset.2 hj))
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      η : Type u_6
      s : η → Set M
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), LinearIndependent R fun x => ↑x
      hη : Not (Nonempty η)
      ⊢ LinearIndependent R fun x => ↑x
    -/
  · refine (linearIndependent_empty R M).mono (t := iUnion (s ·)) ?_
    /-
      case neg
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      η : Type u_6
      s : η → Set M
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), LinearIndependent R fun x => ↑x
      hη : Not (Nonempty η)
      ⊢ HasSubset.Subset (Set.iUnion fun x => s x) EmptyCollection.emptyCollection
    -/
    rintro _ ⟨_, ⟨i, _⟩, _⟩
    /-
      case neg.intro.intro.intro
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      η : Type u_6
      s : η → Set M
      hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
      h : ∀ (i : η), LinearIndependent R fun x => ↑x
      hη : Not (Nonempty η)
      a✝ : M
      w✝ : Set M
      right✝ : Membership.mem w✝ a✝
      i : η
      h✝ : Eq ((fun x => s x) i) w✝
      ⊢ Membership.mem EmptyCollection.emptyCollection a✝
    -/
    exact hη ⟨i⟩
    /-
      🎉 no goals
    -/


theorem linearIndependent_sUnion_of_directed {s : Set (Set M)} (hs : DirectedOn (· ⊆ ·) s)
    (h : ∀ a ∈ s, LinearIndependent R ((↑) : ((a : Set M) : Type _) → M)) :
    LinearIndependent R (fun x => x : ⋃₀ s → M) := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set (Set M)
    hs : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (a : Set M), Membership.mem s a → LinearIndependent R Subtype.val
    ⊢ LinearIndependent R fun x => ↑x
  -/
  rw [sUnion_eq_iUnion]
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set (Set M)
    hs : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (a : Set M), Membership.mem s a → LinearIndependent R Subtype.val
    ⊢ LinearIndependent R fun x => ↑x
  -/
  exact linearIndependent_iUnion_of_directed hs.directed_val (by simpa using h)
  /-
    🎉 no goals
  -/


theorem linearIndependent_biUnion_of_directed {η} {s : Set η} {t : η → Set M}
    (hs : DirectedOn (t ⁻¹'o (· ⊆ ·)) s) (h : ∀ a ∈ s, LinearIndependent R (fun x => x : t a → M)) :
    LinearIndependent R (fun x => x : (⋃ a ∈ s, t a) → M) := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    η : Type u_6
    s : Set η
    t : η → Set M
    hs : DirectedOn (Order.Preimage t fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (a : η), Membership.mem s a → LinearIndependent R fun x => ↑x
    ⊢ LinearIndependent R fun x => ↑x
  -/
  rw [biUnion_eq_iUnion]
  exact
    linearIndependent_iUnion_of_directed (directed_comp.2 <| hs.directed_val) (by simpa using h)


@[deprecated (since := "2024-08-29")] alias linearIndependent_iff_injective_total :=
  linearIndependent_iff_injective_linearCombination


@[deprecated (since := "2024-08-29")] alias LinearIndependent.injective_total :=
  LinearIndependent.injective_linearCombination


theorem LinearIndependent.injective [Nontrivial R] (hv : LinearIndependent R v) : Injective v := by
  simpa [Function.comp_def]
    using Function.Injective.comp hv (Finsupp.single_left_injective one_ne_zero)


theorem LinearIndependent.to_subtype_range {ι} {f : ι → M} (hf : LinearIndependent R f) :
    LinearIndependent R ((↑) : range f → M) := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_6
    f : ι → M
    hf : LinearIndependent R f
    ⊢ LinearIndependent R Subtype.val
  -/
  nontriviality R
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_6
    f : ι → M
    hf : LinearIndependent R f
    a✝ : Nontrivial R
    ⊢ LinearIndependent R Subtype.val
  -/
  exact (linearIndependent_subtype_range hf.injective).2 hf
  /-
    🎉 no goals
  -/


theorem LinearIndependent.to_subtype_range' {ι} {f : ι → M} (hf : LinearIndependent R f) {t}
    (ht : range f = t) : LinearIndependent R ((↑) : t → M) :=
  ht ▸ hf.to_subtype_range


theorem LinearIndependent.image_of_comp {ι ι'} (s : Set ι) (f : ι → ι') (g : ι' → M)
    (hs : LinearIndependent R fun x : s => g (f x)) :
    LinearIndependent R fun x : f '' s => g x := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_6
    ι' : Type u_7
    s : Set ι
    f : ι → ι'
    g : ι' → M
    hs : LinearIndependent R fun x => g (f ↑x)
    ⊢ LinearIndependent R fun x => g ↑x
  -/
  nontriviality R
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_6
    ι' : Type u_7
    s : Set ι
    f : ι → ι'
    g : ι' → M
    hs : LinearIndependent R fun x => g (f ↑x)
    a✝ : Nontrivial R
    ⊢ LinearIndependent R fun x => g ↑x
  -/
  have : InjOn f s := injOn_iff_injective.2 hs.injective.of_comp
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_6
    ι' : Type u_7
    s : Set ι
    f : ι → ι'
    g : ι' → M
    hs : LinearIndependent R fun x => g (f ↑x)
    a✝ : Nontrivial R
    this : Set.InjOn f s
    ⊢ LinearIndependent R fun x => g ↑x
  -/
  exact (linearIndependent_equiv' (Equiv.Set.imageOfInjOn f s this) rfl).1 hs
  /-
    🎉 no goals
  -/


theorem LinearIndependent.image {ι} {s : Set ι} {f : ι → M}
    (hs : LinearIndependent R fun x : s => f x) :
    LinearIndependent R fun x : f '' s => (x : M) := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_6
    s : Set ι
    f : ι → M
    hs : LinearIndependent R fun x => f ↑x
    ⊢ LinearIndependent R fun x => ↑x
  -/
  convert LinearIndependent.image_of_comp s f id hs
  /-
    🎉 no goals
  -/


theorem LinearIndependent.group_smul {G : Type*} [hG : Group G] [DistribMulAction G R]
    [DistribMulAction G M] [IsScalarTower G R M] [SMulCommClass G R M] {v : ι → M}
    (hv : LinearIndependent R v) (w : ι → G) : LinearIndependent R (w • v) := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    G : Type u_6
    hG : Group G
    inst✝³ : DistribMulAction G R
    inst✝² : DistribMulAction G M
    inst✝¹ : IsScalarTower G R M
    inst✝ : SMulCommClass G R M
    v : ι → M
    hv : LinearIndependent R v
    w : ι → G
    ⊢ LinearIndependent R (HSMul.hSMul w v)
  -/
  rw [linearIndependent_iff''] at hv ⊢
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    G : Type u_6
    hG : Group G
    inst✝³ : DistribMulAction G R
    inst✝² : DistribMulAction G M
    inst✝¹ : IsScalarTower G R M
    inst✝ : SMulCommClass G R M
    v : ι → M
    hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
    w : ι → G
    ⊢ ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq (g i …
  -/
  intro s g hgs hsum i
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    G : Type u_6
    hG : Group G
    inst✝³ : DistribMulAction G R
    inst✝² : DistribMulAction G M
    inst✝¹ : IsScalarTower G R M
    inst✝ : SMulCommClass G R M
    v : ι → M
    hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
    w : ι → G
    s : Finset ι
    g : ι → R
    hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
    hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
    i : ι
    ⊢ Eq (g i) 0
  -/
  refine (smul_eq_zero_iff_eq (w i)).1 ?_
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    G : Type u_6
    hG : Group G
    inst✝³ : DistribMulAction G R
    inst✝² : DistribMulAction G M
    inst✝¹ : IsScalarTower G R M
    inst✝ : SMulCommClass G R M
    v : ι → M
    hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
    w : ι → G
    s : Finset ι
    g : ι → R
    hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
    hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
    i : ι
    ⊢ Eq (HSMul.hSMul (w i) (g i)) 0
  -/
  refine hv s (fun i => w i • g i) (fun i hi => ?_) ?_ i
    /-
      case refine_1
      ι : Type u'
      R : Type u_2
      M : Type u_4
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      G : Type u_6
      hG : Group G
      inst✝³ : DistribMulAction G R
      inst✝² : DistribMulAction G M
      inst✝¹ : IsScalarTower G R M
      inst✝ : SMulCommClass G R M
      v : ι → M
      hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
      w : ι → G
      s : Finset ι
      g : ι → R
      hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
      hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
      i✝ i : ι
      hi : Not (Membership.mem s i)
      ⊢ Eq ((fun i => HSMul.hSMul (w i) (g i)) i) 0
    -/
  · dsimp only
    /-
      case refine_1
      ι : Type u'
      R : Type u_2
      M : Type u_4
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      G : Type u_6
      hG : Group G
      inst✝³ : DistribMulAction G R
      inst✝² : DistribMulAction G M
      inst✝¹ : IsScalarTower G R M
      inst✝ : SMulCommClass G R M
      v : ι → M
      hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
      w : ι → G
      s : Finset ι
      g : ι → R
      hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
      hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
      i✝ i : ι
      hi : Not (Membership.mem s i)
      ⊢ Eq (HSMul.hSMul (w i) (g i)) 0
    -/
    exact (hgs i hi).symm ▸ smul_zero _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u'
      R : Type u_2
      M : Type u_4
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      G : Type u_6
      hG : Group G
      inst✝³ : DistribMulAction G R
      inst✝² : DistribMulAction G M
      inst✝¹ : IsScalarTower G R M
      inst✝ : SMulCommClass G R M
      v : ι → M
      hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
      w : ι → G
      s : Finset ι
      g : ι → R
      hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
      hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
      i : ι
      ⊢ Eq (s.sum fun i => HSMul.hSMul ((fun i => HSMul.hSMul (w i) (g i)) i) (v i)) 0
    -/
  · rw [← hsum, Finset.sum_congr rfl _]
    /-
      ι : Type u'
      R : Type u_2
      M : Type u_4
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      G : Type u_6
      hG : Group G
      inst✝³ : DistribMulAction G R
      inst✝² : DistribMulAction G M
      inst✝¹ : IsScalarTower G R M
      inst✝ : SMulCommClass G R M
      v : ι → M
      hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
      w : ι → G
      s : Finset ι
      g : ι → R
      hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
      hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
      i : ι
      ⊢ ∀ (x : ι), Membership.mem s x → Eq (HSMul.hSMul ((fun i => HSMul.hSMul (w i) …
    -/
    intros
    /-
      ι : Type u'
      R : Type u_2
      M : Type u_4
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      G : Type u_6
      hG : Group G
      inst✝³ : DistribMulAction G R
      inst✝² : DistribMulAction G M
      inst✝¹ : IsScalarTower G R M
      inst✝ : SMulCommClass G R M
      v : ι → M
      hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
      w : ι → G
      s : Finset ι
      g : ι → R
      hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
      hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
      i x✝ : ι
      a✝ : Membership.mem s x✝
      ⊢ Eq (HSMul.hSMul ((fun i => HSMul.hSMul (w i) (g i)) x✝) (v x✝)) (HSMul.hSMul …
    -/
    dsimp
    /-
      ι : Type u'
      R : Type u_2
      M : Type u_4
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      G : Type u_6
      hG : Group G
      inst✝³ : DistribMulAction G R
      inst✝² : DistribMulAction G M
      inst✝¹ : IsScalarTower G R M
      inst✝ : SMulCommClass G R M
      v : ι → M
      hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
      w : ι → G
      s : Finset ι
      g : ι → R
      hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
      hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
      i x✝ : ι
      a✝ : Membership.mem s x✝
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul (w x✝) (g x✝)) (v x✝)) (HSMul.hSMul (g x✝) (HSM …
    -/
    rw [smul_assoc, smul_comm]
    /-
      🎉 no goals
    -/

-- This lemma cannot be proved with `LinearIndependent.group_smul` since the action of
-- `Rˣ` on `R` is not commutative.

theorem LinearIndependent.units_smul {v : ι → M} (hv : LinearIndependent R v) (w : ι → Rˣ) :
    LinearIndependent R (w • v) := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v : ι → M
    hv : LinearIndependent R v
    w : ι → Units R
    ⊢ LinearIndependent R (HSMul.hSMul w v)
  -/
  rw [linearIndependent_iff''] at hv ⊢
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v : ι → M
    hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
    w : ι → Units R
    ⊢ ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq (g i …
  -/
  intro s g hgs hsum i
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v : ι → M
    hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
    w : ι → Units R
    s : Finset ι
    g : ι → R
    hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
    hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
    i : ι
    ⊢ Eq (g i) 0
  -/
  rw [← (w i).mul_left_eq_zero]
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v : ι → M
    hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
    w : ι → Units R
    s : Finset ι
    g : ι → R
    hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
    hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
    i : ι
    ⊢ Eq (HMul.hMul (g i) ↑(w i)) 0
  -/
  refine hv s (fun i => g i • (w i : R)) (fun i hi => ?_) ?_ i
    /-
      case refine_1
      ι : Type u'
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
      w : ι → Units R
      s : Finset ι
      g : ι → R
      hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
      hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
      i✝ i : ι
      hi : Not (Membership.mem s i)
      ⊢ Eq ((fun i => HSMul.hSMul (g i) ↑(w i)) i) 0
    -/
  · dsimp only
    /-
      case refine_1
      ι : Type u'
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
      w : ι → Units R
      s : Finset ι
      g : ι → R
      hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
      hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
      i✝ i : ι
      hi : Not (Membership.mem s i)
      ⊢ Eq (HSMul.hSMul (g i) ↑(w i)) 0
    -/
    exact (hgs i hi).symm ▸ zero_smul _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u'
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
      w : ι → Units R
      s : Finset ι
      g : ι → R
      hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
      hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
      i : ι
      ⊢ Eq (s.sum fun i => HSMul.hSMul ((fun i => HSMul.hSMul (g i) ↑(w i)) i) (v i) …
    -/
  · rw [← hsum, Finset.sum_congr rfl _]
    /-
      ι : Type u'
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
      w : ι → Units R
      s : Finset ι
      g : ι → R
      hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
      hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
      i : ι
      ⊢ ∀ (x : ι), Membership.mem s x → Eq (HSMul.hSMul ((fun i => HSMul.hSMul (g i) …
    -/
    intros
    /-
      ι : Type u'
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      hv : ∀ (s : Finset ι) (g : ι → R), (∀ (i : ι), Not (Membership.mem s i) → Eq ( …
      w : ι → Units R
      s : Finset ι
      g : ι → R
      hgs : ∀ (i : ι), Not (Membership.mem s i) → Eq (g i) 0
      hsum : Eq (s.sum fun i => HSMul.hSMul (g i) (HSMul.hSMul w v i)) 0
      i x✝ : ι
      a✝ : Membership.mem s x✝
      ⊢ Eq (HSMul.hSMul ((fun i => HSMul.hSMul (g i) ↑(w i)) x✝) (v x✝)) (HSMul.hSMu …
    -/
    rw [Pi.smul_apply', smul_assoc, Units.smul_def]
    /-
      🎉 no goals
    -/


lemma LinearIndependent.eq_of_pair {x y : M} (h : LinearIndependent R ![x, y])
    {s t s' t' : R} (h' : s • x + t • y = s' • x + t' • y) : s = s' ∧ t = t' := by
  have : (s - s') • x + (t - t') • y = 0 := by
    rw [← sub_eq_zero_of_eq h']
    match_scalars <;> noncomm_ring
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    s t s' t' : R
    h' : Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) (HAdd.hAdd (HSMul.hSMu …
    this : Eq (HAdd.hAdd (HSMul.hSMul (HSub.hSub s s') x) (HSMul.hSMul (HSub.hSub  …
    ⊢ And (Eq s s') (Eq t t')
  -/
  simpa [sub_eq_zero] using h.eq_zero_of_pair this
  /-
    🎉 no goals
  -/


lemma LinearIndependent.eq_zero_of_pair' {x y : M} (h : LinearIndependent R ![x, y])
    {s t : R} (h' : s • x = t • y) : s = 0 ∧ t = 0 := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    s t : R
    h' : Eq (HSMul.hSMul s x) (HSMul.hSMul t y)
    ⊢ And (Eq s 0) (Eq t 0)
  -/
  suffices H : s = 0 ∧ 0 = t from ⟨H.1, H.2.symm⟩
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    s t : R
    h' : Eq (HSMul.hSMul s x) (HSMul.hSMul t y)
    ⊢ And (Eq s 0) (Eq 0 t)
  -/
  exact h.eq_of_pair (by simpa using h')
  /-
    🎉 no goals
  -/


/-- If two vectors `x` and `y` are linearly independent, so are their linear combinations
`a x + b y` and `c x + d y` provided the determinant `a * d - b * c` is nonzero. -/
lemma LinearIndependent.linear_combination_pair_of_det_ne_zero {R M : Type*} [CommRing R]
    [NoZeroDivisors R] [AddCommGroup M] [Module R M]
    {x y : M} (h : LinearIndependent R ![x, y])
    {a b c d : R} (h' : a * d - b * c ≠ 0) :
    LinearIndependent R ![a • x + b • y, c • x + d • y] := by
  /-
    R : Type u_6
    M : Type u_7
    inst✝³ : CommRing R
    inst✝² : NoZeroDivisors R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    a b c d : R
    h' : Ne (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 0
    ⊢ LinearIndependent R (Matrix.vecCons (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMu …
  -/
  apply LinearIndependent.pair_iff.2 (fun s t hst ↦ ?_)
  have H : (s * a + t * c) • x + (s * b + t * d) • y = 0 := by
    convert hst using 1
    module
  /-
    R : Type u_6
    M : Type u_7
    inst✝³ : CommRing R
    inst✝² : NoZeroDivisors R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    a b c d : R
    h' : Ne (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 0
    s t : R
    hst : Eq (HAdd.hAdd (HSMul.hSMul s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b …
    H : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (HMul.hMul s a) (HMul.hMul t c)) x)  …
    ⊢ And (Eq s 0) (Eq t 0)
  -/
  have I1 : s * a + t * c = 0 := (h.eq_zero_of_pair H).1
  /-
    R : Type u_6
    M : Type u_7
    inst✝³ : CommRing R
    inst✝² : NoZeroDivisors R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    a b c d : R
    h' : Ne (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 0
    s t : R
    hst : Eq (HAdd.hAdd (HSMul.hSMul s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b …
    H : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (HMul.hMul s a) (HMul.hMul t c)) x)  …
    I1 : Eq (HAdd.hAdd (HMul.hMul s a) (HMul.hMul t c)) 0
    ⊢ And (Eq s 0) (Eq t 0)
  -/
  have I2 : s * b + t * d = 0 := (h.eq_zero_of_pair H).2
  /-
    R : Type u_6
    M : Type u_7
    inst✝³ : CommRing R
    inst✝² : NoZeroDivisors R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    a b c d : R
    h' : Ne (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 0
    s t : R
    hst : Eq (HAdd.hAdd (HSMul.hSMul s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b …
    H : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (HMul.hMul s a) (HMul.hMul t c)) x)  …
    I1 : Eq (HAdd.hAdd (HMul.hMul s a) (HMul.hMul t c)) 0
    I2 : Eq (HAdd.hAdd (HMul.hMul s b) (HMul.hMul t d)) 0
    ⊢ And (Eq s 0) (Eq t 0)
  -/
  have J1 : (a * d - b * c) * s = 0 := by linear_combination d * I1 - c * I2
  /-
    R : Type u_6
    M : Type u_7
    inst✝³ : CommRing R
    inst✝² : NoZeroDivisors R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    a b c d : R
    h' : Ne (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 0
    s t : R
    hst : Eq (HAdd.hAdd (HSMul.hSMul s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b …
    H : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (HMul.hMul s a) (HMul.hMul t c)) x)  …
    I1 : Eq (HAdd.hAdd (HMul.hMul s a) (HMul.hMul t c)) 0
    I2 : Eq (HAdd.hAdd (HMul.hMul s b) (HMul.hMul t d)) 0
    J1 : Eq (HMul.hMul (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) s) 0
    ⊢ And (Eq s 0) (Eq t 0)
  -/
  have J2 : (a * d - b * c) * t = 0 := by linear_combination -b * I1 + a * I2
  /-
    R : Type u_6
    M : Type u_7
    inst✝³ : CommRing R
    inst✝² : NoZeroDivisors R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    a b c d : R
    h' : Ne (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 0
    s t : R
    hst : Eq (HAdd.hAdd (HSMul.hSMul s (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b …
    H : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (HMul.hMul s a) (HMul.hMul t c)) x)  …
    I1 : Eq (HAdd.hAdd (HMul.hMul s a) (HMul.hMul t c)) 0
    I2 : Eq (HAdd.hAdd (HMul.hMul s b) (HMul.hMul t d)) 0
    J1 : Eq (HMul.hMul (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) s) 0
    J2 : Eq (HMul.hMul (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) t) 0
    ⊢ And (Eq s 0) (Eq t 0)
  -/
  exact ⟨by simpa [h'] using mul_eq_zero.1 J1, by simpa [h'] using mul_eq_zero.1 J2⟩
  /-
    🎉 no goals
  -/


/--
A linearly independent family is maximal if there is no strictly larger linearly independent family.
-/
@[nolint unusedArguments]
def LinearIndependent.Maximal {ι : Type w} {R : Type u} [Ring R] {M : Type v} [AddCommGroup M]
    [Module R M] {v : ι → M} (_i : LinearIndependent R v) : Prop :=
  ∀ (s : Set M) (_i' : LinearIndependent R ((↑) : s → M)) (_h : range v ≤ s), range v = s


/-- An alternative characterization of a maximal linearly independent family,
quantifying over types (in the same universe as `M`) into which the indexing family injects.
-/
theorem LinearIndependent.maximal_iff {ι : Type w} {R : Type u} [Ring R] [Nontrivial R] {M : Type v}
    [AddCommGroup M] [Module R M] {v : ι → M} (i : LinearIndependent R v) :
    i.Maximal ↔
      ∀ (κ : Type v) (w : κ → M) (_i' : LinearIndependent R w) (j : ι → κ) (_h : w ∘ j = v),
        Surjective j := by
  /-
    ι : Type w
    R : Type u
    inst✝³ : Ring R
    inst✝² : Nontrivial R
    M : Type v
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v : ι → M
    i : LinearIndependent R v
    ⊢ Iff i.Maximal (∀ (κ : Type v) (w : κ → M), LinearIndependent R w → ∀ (j : ι  …
  -/
  constructor
    /-
      case mp
      ι : Type w
      R : Type u
      inst✝³ : Ring R
      inst✝² : Nontrivial R
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      i : LinearIndependent R v
      ⊢ i.Maximal → ∀ (κ : Type v) (w : κ → M), LinearIndependent R w → ∀ (j : ι → κ …
    -/
  · rintro p κ w i' j rfl
    /-
      case mp
      ι : Type w
      R : Type u
      inst✝³ : Ring R
      inst✝² : Nontrivial R
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      κ : Type v
      w : κ → M
      i' : LinearIndependent R w
      j : ι → κ
      i : LinearIndependent R (Function.comp w j)
      p : i.Maximal
      ⊢ Function.Surjective j
    -/
    specialize p (range w) i'.coe_range (range_comp_subset_range _ _)
    /-
      case mp
      ι : Type w
      R : Type u
      inst✝³ : Ring R
      inst✝² : Nontrivial R
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      κ : Type v
      w : κ → M
      i' : LinearIndependent R w
      j : ι → κ
      i : LinearIndependent R (Function.comp w j)
      p : Eq (Set.range (Function.comp w j)) (Set.range w)
      ⊢ Function.Surjective j
    -/
    rw [range_comp, ← image_univ (f := w)] at p
    /-
      case mp
      ι : Type w
      R : Type u
      inst✝³ : Ring R
      inst✝² : Nontrivial R
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      κ : Type v
      w : κ → M
      i' : LinearIndependent R w
      j : ι → κ
      i : LinearIndependent R (Function.comp w j)
      p : Eq (Set.image w (Set.range j)) (Set.image w Set.univ)
      ⊢ Function.Surjective j
    -/
    exact range_eq_univ.mp (image_injective.mpr i'.injective p)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type w
      R : Type u
      inst✝³ : Ring R
      inst✝² : Nontrivial R
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      i : LinearIndependent R v
      ⊢ (∀ (κ : Type v) (w : κ → M), LinearIndependent R w → ∀ (j : ι → κ), Eq (Func …
    -/
  · intro p w i' h
    specialize
      p w ((↑) : w → M) i' (fun i => ⟨v i, range_subset_iff.mp h i⟩)
        (by
          ext
          simp)
    /-
      case mpr
      ι : Type w
      R : Type u
      inst✝³ : Ring R
      inst✝² : Nontrivial R
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      i' : LinearIndependent R Subtype.val
      h : LE.le (Set.range v) w
      p : Function.Surjective fun i => ⟨v i, ⋯⟩
      ⊢ Eq (Set.range v) w
    -/
    have q := congr_arg (fun s => ((↑) : w → M) '' s) p.range_eq
    /-
      case mpr
      ι : Type w
      R : Type u
      inst✝³ : Ring R
      inst✝² : Nontrivial R
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      i' : LinearIndependent R Subtype.val
      h : LE.le (Set.range v) w
      p : Function.Surjective fun i => ⟨v i, ⋯⟩
      q : Eq ((fun s => Set.image Subtype.val s) (Set.range fun i => ⟨v i, ⋯⟩)) ((fu …
      ⊢ Eq (Set.range v) w
    -/
    dsimp at q
    /-
      case mpr
      ι : Type w
      R : Type u
      inst✝³ : Ring R
      inst✝² : Nontrivial R
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      i' : LinearIndependent R Subtype.val
      h : LE.le (Set.range v) w
      p : Function.Surjective fun i => ⟨v i, ⋯⟩
      q : Eq (Set.image Subtype.val (Set.range fun i => ⟨v i, ⋯⟩)) (Set.image Subtyp …
      ⊢ Eq (Set.range v) w
    -/
    rw [← image_univ, image_image] at q
    /-
      case mpr
      ι : Type w
      R : Type u
      inst✝³ : Ring R
      inst✝² : Nontrivial R
      M : Type v
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      i : LinearIndependent R v
      w : Set M
      i' : LinearIndependent R Subtype.val
      h : LE.le (Set.range v) w
      p : Function.Surjective fun i => ⟨v i, ⋯⟩
      q : Eq (Set.image (fun x => ↑⟨v x, ⋯⟩) Set.univ) (Set.image Subtype.val Set.un …
      ⊢ Eq (Set.range v) w
    -/
    simpa using q
    /-
      🎉 no goals
    -/


/-- Linear independent families are injective, even if you multiply either side. -/
theorem LinearIndependent.eq_of_smul_apply_eq_smul_apply {M : Type*} [AddCommGroup M] [Module R M]
    {v : ι → M} (li : LinearIndependent R v) (c d : R) (i j : ι) (hc : c ≠ 0)
    (h : c • v i = d • v j) : i = j := by
  /-
    ι : Type u'
    R : Type u_2
    inst✝² : Ring R
    M : Type u_6
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v : ι → M
    li : LinearIndependent R v
    c d : R
    i j : ι
    hc : Ne c 0
    h : Eq (HSMul.hSMul c (v i)) (HSMul.hSMul d (v j))
    ⊢ Eq i j
  -/
  let l : ι →₀ R := Finsupp.single i c - Finsupp.single j d
  have h_total : Finsupp.linearCombination R v l = 0 := by
    simp_rw [l, LinearMap.map_sub, Finsupp.linearCombination_apply]
    simp [h]
  have h_single_eq : Finsupp.single i c = Finsupp.single j d := by
    rw [linearIndependent_iff] at li
    simp [eq_add_of_sub_eq' (li l h_total)]
  /-
    ι : Type u'
    R : Type u_2
    inst✝² : Ring R
    M : Type u_6
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v : ι → M
    li : LinearIndependent R v
    c d : R
    i j : ι
    hc : Ne c 0
    h : Eq (HSMul.hSMul c (v i)) (HSMul.hSMul d (v j))
    l : Finsupp ι R := HSub.hSub (Finsupp.single i c) (Finsupp.single j d)
    h_total : Eq ((Finsupp.linearCombination R v) l) 0
    h_single_eq : Eq (Finsupp.single i c) (Finsupp.single j d)
    ⊢ Eq i j
  -/
  rcases (Finsupp.single_eq_single_iff ..).mp h_single_eq with (⟨H, _⟩ | ⟨hc, _⟩)
    /-
      case inl.intro
      ι : Type u'
      R : Type u_2
      inst✝² : Ring R
      M : Type u_6
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      li : LinearIndependent R v
      c d : R
      i j : ι
      hc : Ne c 0
      h : Eq (HSMul.hSMul c (v i)) (HSMul.hSMul d (v j))
      l : Finsupp ι R := HSub.hSub (Finsupp.single i c) (Finsupp.single j d)
      h_total : Eq ((Finsupp.linearCombination R v) l) 0
      h_single_eq : Eq (Finsupp.single i c) (Finsupp.single j d)
      H : Eq i j
      right✝ : Eq c d
      ⊢ Eq i j
    -/
  · exact H
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      ι : Type u'
      R : Type u_2
      inst✝² : Ring R
      M : Type u_6
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      v : ι → M
      li : LinearIndependent R v
      c d : R
      i j : ι
      hc✝ : Ne c 0
      h : Eq (HSMul.hSMul c (v i)) (HSMul.hSMul d (v j))
      l : Finsupp ι R := HSub.hSub (Finsupp.single i c) (Finsupp.single j d)
      h_total : Eq ((Finsupp.linearCombination R v) l) 0
      h_single_eq : Eq (Finsupp.single i c) (Finsupp.single j d)
      hc : Eq c 0
      right✝ : Eq d 0
      ⊢ Eq i j
    -/
  · contradiction
    /-
      🎉 no goals
    -/


theorem LinearIndependent.disjoint_span_image (hv : LinearIndependent R v) {s t : Set ι}
    (hs : Disjoint s t) : Disjoint (Submodule.span R <| v '' s) (Submodule.span R <| v '' t) := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    s t : Set ι
    hs : Disjoint s t
    ⊢ Disjoint (Submodule.span R (Set.image v s)) (Submodule.span R (Set.image v t))
  -/
  simp only [disjoint_def, Finsupp.mem_span_image_iff_linearCombination]
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    s t : Set ι
    hs : Disjoint s t
    ⊢ ∀ (x : M), (Exists fun l => And (Membership.mem (Finsupp.supported R R s) l) …
  -/
  rintro _ ⟨l₁, hl₁, rfl⟩ ⟨l₂, hl₂, H⟩
  /-
    case intro.intro.intro.intro
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    s t : Set ι
    hs : Disjoint s t
    l₁ : Finsupp ι R
    hl₁ : Membership.mem (Finsupp.supported R R s) l₁
    l₂ : Finsupp ι R
    hl₂ : Membership.mem (Finsupp.supported R R t) l₂
    H : Eq ((Finsupp.linearCombination R v) l₂) ((Finsupp.linearCombination R v) l₁)
    ⊢ Eq ((Finsupp.linearCombination R v) l₁) 0
  -/
  rw [hv.injective_linearCombination.eq_iff] at H; subst l₂
  /-
    case intro.intro.intro.intro
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    s t : Set ι
    hs : Disjoint s t
    l₁ : Finsupp ι R
    hl₁ : Membership.mem (Finsupp.supported R R s) l₁
    hl₂ : Membership.mem (Finsupp.supported R R t) l₁
    ⊢ Eq ((Finsupp.linearCombination R v) l₁) 0
  -/
  have : l₁ = 0 := Submodule.disjoint_def.mp (Finsupp.disjoint_supported_supported hs) _ hl₁ hl₂
  /-
    case intro.intro.intro.intro
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    s t : Set ι
    hs : Disjoint s t
    l₁ : Finsupp ι R
    hl₁ : Membership.mem (Finsupp.supported R R s) l₁
    hl₂ : Membership.mem (Finsupp.supported R R t) l₁
    this : Eq l₁ 0
    ⊢ Eq ((Finsupp.linearCombination R v) l₁) 0
  -/
  simp [this]
  /-
    🎉 no goals
  -/


theorem LinearIndependent.not_mem_span_image [Nontrivial R] (hv : LinearIndependent R v) {s : Set ι}
    {x : ι} (h : x ∉ s) : v x ∉ Submodule.span R (v '' s) := by
  have h' : v x ∈ Submodule.span R (v '' {x}) := by
    rw [Set.image_singleton]
    exact mem_span_singleton_self (v x)
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    s : Set ι
    x : ι
    h : Not (Membership.mem s x)
    h' : Membership.mem (Submodule.span R (Set.image v (Singleton.singleton x))) ( …
    ⊢ Not (Membership.mem (Submodule.span R (Set.image v s)) (v x))
  -/
  intro w
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    s : Set ι
    x : ι
    h : Not (Membership.mem s x)
    h' : Membership.mem (Submodule.span R (Set.image v (Singleton.singleton x))) ( …
    w : Membership.mem (Submodule.span R (Set.image v s)) (v x)
    ⊢ False
  -/
  apply LinearIndependent.ne_zero x hv
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    s : Set ι
    x : ι
    h : Not (Membership.mem s x)
    h' : Membership.mem (Submodule.span R (Set.image v (Singleton.singleton x))) ( …
    w : Membership.mem (Submodule.span R (Set.image v s)) (v x)
    ⊢ Eq (v x) 0
  -/
  refine disjoint_def.1 (hv.disjoint_span_image ?_) (v x) h' w
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    s : Set ι
    x : ι
    h : Not (Membership.mem s x)
    h' : Membership.mem (Submodule.span R (Set.image v (Singleton.singleton x))) ( …
    w : Membership.mem (Submodule.span R (Set.image v s)) (v x)
    ⊢ Disjoint (Singleton.singleton x) s
  -/
  simpa using h
  /-
    🎉 no goals
  -/


theorem LinearIndependent.linearCombination_ne_of_not_mem_support [Nontrivial R]
    (hv : LinearIndependent R v) {x : ι} (f : ι →₀ R) (h : x ∉ f.support) :
    Finsupp.linearCombination R v f ≠ v x := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    x : ι
    f : Finsupp ι R
    h : Not (Membership.mem f.support x)
    ⊢ Ne ((Finsupp.linearCombination R v) f) (v x)
  -/
  replace h : x ∉ (f.support : Set ι) := h
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    x : ι
    f : Finsupp ι R
    h : Not (Membership.mem (↑f.support) x)
    ⊢ Ne ((Finsupp.linearCombination R v) f) (v x)
  -/
  have p := hv.not_mem_span_image h
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    x : ι
    f : Finsupp ι R
    h : Not (Membership.mem (↑f.support) x)
    p : Not (Membership.mem (Submodule.span R (Set.image v ↑f.support)) (v x))
    ⊢ Ne ((Finsupp.linearCombination R v) f) (v x)
  -/
  intro w
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    x : ι
    f : Finsupp ι R
    h : Not (Membership.mem (↑f.support) x)
    p : Not (Membership.mem (Submodule.span R (Set.image v ↑f.support)) (v x))
    w : Eq ((Finsupp.linearCombination R v) f) (v x)
    ⊢ False
  -/
  rw [← w] at p
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    x : ι
    f : Finsupp ι R
    h : Not (Membership.mem (↑f.support) x)
    p : Not (Membership.mem (Submodule.span R (Set.image v ↑f.support)) ((Finsupp. …
    w : Eq ((Finsupp.linearCombination R v) f) (v x)
    ⊢ False
  -/
  rw [Finsupp.span_image_eq_map_linearCombination] at p
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    x : ι
    f : Finsupp ι R
    h : Not (Membership.mem (↑f.support) x)
    p : Not (Membership.mem (Submodule.map (Finsupp.linearCombination R v) (Finsup …
    w : Eq ((Finsupp.linearCombination R v) f) (v x)
    ⊢ False
  -/
  simp only [not_exists, not_and, mem_map] at p -- Porting note: `mem_map` isn't currently triggered
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    x : ι
    f : Finsupp ι R
    h : Not (Membership.mem (↑f.support) x)
    w : Eq ((Finsupp.linearCombination R v) f) (v x)
    p : ∀ (x : Finsupp ι R), Membership.mem (Finsupp.supported R R ↑f.support) x → …
    ⊢ False
  -/
  exact p f (f.mem_supported_support R) rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias LinearIndependent.total_ne_of_not_mem_support :=
  LinearIndependent.linearCombination_ne_of_not_mem_support


theorem linearIndependent_sum {v : ι ⊕ ι' → M} :
    LinearIndependent R v ↔
      LinearIndependent R (v ∘ Sum.inl) ∧
        LinearIndependent R (v ∘ Sum.inr) ∧
          Disjoint (Submodule.span R (range (v ∘ Sum.inl)))
            (Submodule.span R (range (v ∘ Sum.inr))) := by
  classical
  rw [range_comp v, range_comp v]
  refine ⟨?_, ?_⟩
  · intro h
    refine ⟨h.comp _ Sum.inl_injective, h.comp _ Sum.inr_injective, ?_⟩
    refine h.disjoint_span_image ?_
    -- Porting note: `isCompl_range_inl_range_inr.1` timeouts.
    exact IsCompl.disjoint isCompl_range_inl_range_inr
  rintro ⟨hl, hr, hlr⟩
  rw [linearIndependent_iff'] at *
  intro s g hg i hi
  have :
    ((∑ i ∈ s.preimage Sum.inl Sum.inl_injective.injOn, (fun x => g x • v x) (Sum.inl i)) +
        ∑ i ∈ s.preimage Sum.inr Sum.inr_injective.injOn, (fun x => g x • v x) (Sum.inr i)) =
      0 := by
    -- Porting note: `g` must be specified.
    rw [Finset.sum_preimage' (g := fun x => g x • v x),
      Finset.sum_preimage' (g := fun x => g x • v x), ← Finset.sum_union, ← Finset.filter_or]
    · simpa only [← mem_union, range_inl_union_range_inr, mem_univ, Finset.filter_True]
    · -- Porting note: Here was one `exact`, but timeouted.
      refine Finset.disjoint_filter.2 fun x _ hx =>
        disjoint_left.1 ?_ hx
      exact IsCompl.disjoint isCompl_range_inl_range_inr
  rw [← eq_neg_iff_add_eq_zero] at this
  rw [disjoint_def'] at hlr
  have A := by
    refine hlr _ (sum_mem fun i _ => ?_) _ (neg_mem <| sum_mem fun i _ => ?_) this
    · exact smul_mem _ _ (subset_span ⟨Sum.inl i, mem_range_self _, rfl⟩)
    · exact smul_mem _ _ (subset_span ⟨Sum.inr i, mem_range_self _, rfl⟩)
  cases' i with i i
  · exact hl _ _ A i (Finset.mem_preimage.2 hi)
  · rw [this, neg_eq_zero] at A
    exact hr _ _ A i (Finset.mem_preimage.2 hi)


theorem LinearIndependent.sum_type {v' : ι' → M} (hv : LinearIndependent R v)
    (hv' : LinearIndependent R v')
    (h : Disjoint (Submodule.span R (range v)) (Submodule.span R (range v'))) :
    LinearIndependent R (Sum.elim v v') :=
  linearIndependent_sum.2 ⟨hv, hv', h⟩


theorem LinearIndependent.union {s t : Set M} (hs : LinearIndependent R (fun x => x : s → M))
    (ht : LinearIndependent R (fun x => x : t → M)) (hst : Disjoint (span R s) (span R t)) :
    LinearIndependent R (fun x => x : ↥(s ∪ t) → M) :=
                        /-
                          R : Type u_2
                          M : Type u_4
                          inst✝² : Ring R
                          inst✝¹ : AddCommGroup M
                          inst✝ : Module R M
                          s t : Set M
                          hs : LinearIndependent R fun x => ↑x
                          ht : LinearIndependent R fun x => ↑x
                          hst : Disjoint (Submodule.span R s) (Submodule.span R t)
                          ⊢ Disjoint (Submodule.span R (Set.range fun x => ↑x)) (Submodule.span R (Set.r …
                        -/
                        /-
                          🎉 no goals
                        -/
  (hs.sum_type ht <| by simpa).to_subtype_range' <| by simp
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem linearIndependent_iUnion_finite_subtype {ι : Type*} {f : ι → Set M}
    (hl : ∀ i, LinearIndependent R (fun x => x : f i → M))
    (hd : ∀ i, ∀ t : Set ι, t.Finite → i ∉ t → Disjoint (span R (f i)) (⨆ i ∈ t, span R (f i))) :
    LinearIndependent R (fun x => x : (⋃ i, f i) → M) := by
  classical
  rw [iUnion_eq_iUnion_finset f]
  apply linearIndependent_iUnion_of_directed
  · apply directed_of_isDirected_le
    exact fun t₁ t₂ ht => iUnion_mono fun i => iUnion_subset_iUnion_const fun h => ht h
  intro t
  induction' t using Finset.induction_on with i s his ih
  · refine (linearIndependent_empty R M).mono ?_
    simp
  · rw [Finset.set_biUnion_insert]
    refine (hl _).union ih ?_
    rw [span_iUnion₂]
    exact hd i s s.finite_toSet his


theorem linearIndependent_iUnion_finite {η : Type*} {ιs : η → Type*} {f : ∀ j : η, ιs j → M}
    (hindep : ∀ j, LinearIndependent R (f j))
    (hd : ∀ i, ∀ t : Set η,
      t.Finite → i ∉ t → Disjoint (span R (range (f i))) (⨆ i ∈ t, span R (range (f i)))) :
    LinearIndependent R fun ji : Σ j, ιs j => f ji.1 ji.2 := by
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    η : Type u_6
    ιs : η → Type u_7
    f : (j : η) → ιs j → M
    hindep : ∀ (j : η), LinearIndependent R (f j)
    hd : ∀ (i : η) (t : Set η), t.Finite → Not (Membership.mem t i) → Disjoint (Su …
    ⊢ LinearIndependent R fun ji => f ji.fst ji.snd
  -/
  nontriviality R
  /-
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    η : Type u_6
    ιs : η → Type u_7
    f : (j : η) → ιs j → M
    hindep : ∀ (j : η), LinearIndependent R (f j)
    hd : ∀ (i : η) (t : Set η), t.Finite → Not (Membership.mem t i) → Disjoint (Su …
    a✝ : Nontrivial R
    ⊢ LinearIndependent R fun ji => f ji.fst ji.snd
  -/
  apply LinearIndependent.of_subtype_range
    /-
      case hf
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      η : Type u_6
      ιs : η → Type u_7
      f : (j : η) → ιs j → M
      hindep : ∀ (j : η), LinearIndependent R (f j)
      hd : ∀ (i : η) (t : Set η), t.Finite → Not (Membership.mem t i) → Disjoint (Su …
      a✝ : Nontrivial R
      ⊢ Function.Injective fun ji => f ji.fst ji.snd
    -/
  · rintro ⟨x₁, x₂⟩ ⟨y₁, y₂⟩ hxy
    /-
      case hf.mk.mk
      R : Type u_2
      M : Type u_4
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      η : Type u_6
      ιs : η → Type u_7
      f : (j : η) → ιs j → M
      hindep : ∀ (j : η), LinearIndependent R (f j)
      hd : ∀ (i : η) (t : Set η), t.Finite → Not (Membership.mem t i) → Disjoint (Su …
      a✝ : Nontrivial R
      x₁ : η
      x₂ : ιs x₁
      y₁ : η
      y₂ : ιs y₁
      hxy : Eq ((fun ji => f ji.fst ji.snd) ⟨x₁, x₂⟩) ((fun ji => f ji.fst ji.snd) ⟨ …
      ⊢ Eq ⟨x₁, x₂⟩ ⟨y₁, y₂⟩
    -/
    by_cases h_cases : x₁ = y₁
      /-
        case pos
        R : Type u_2
        M : Type u_4
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        η : Type u_6
        ιs : η → Type u_7
        f : (j : η) → ιs j → M
        hindep : ∀ (j : η), LinearIndependent R (f j)
        hd : ∀ (i : η) (t : Set η), t.Finite → Not (Membership.mem t i) → Disjoint (Su …
        a✝ : Nontrivial R
        x₁ : η
        x₂ : ιs x₁
        y₁ : η
        y₂ : ιs y₁
        hxy : Eq ((fun ji => f ji.fst ji.snd) ⟨x₁, x₂⟩) ((fun ji => f ji.fst ji.snd) ⟨ …
        h_cases : Eq x₁ y₁
        ⊢ Eq ⟨x₁, x₂⟩ ⟨y₁, y₂⟩
      -/
    · subst h_cases
      /-
        case pos
        R : Type u_2
        M : Type u_4
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        η : Type u_6
        ιs : η → Type u_7
        f : (j : η) → ιs j → M
        hindep : ∀ (j : η), LinearIndependent R (f j)
        hd : ∀ (i : η) (t : Set η), t.Finite → Not (Membership.mem t i) → Disjoint (Su …
        a✝ : Nontrivial R
        x₁ : η
        x₂ y₂ : ιs x₁
        hxy : Eq ((fun ji => f ji.fst ji.snd) ⟨x₁, x₂⟩) ((fun ji => f ji.fst ji.snd) ⟨ …
        ⊢ Eq ⟨x₁, x₂⟩ ⟨x₁, y₂⟩
      -/
      refine Sigma.eq rfl ?_
      /-
        case pos
        R : Type u_2
        M : Type u_4
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        η : Type u_6
        ιs : η → Type u_7
        f : (j : η) → ιs j → M
        hindep : ∀ (j : η), LinearIndependent R (f j)
        hd : ∀ (i : η) (t : Set η), t.Finite → Not (Membership.mem t i) → Disjoint (Su …
        a✝ : Nontrivial R
        x₁ : η
        x₂ y₂ : ιs x₁
        hxy : Eq ((fun ji => f ji.fst ji.snd) ⟨x₁, x₂⟩) ((fun ji => f ji.fst ji.snd) ⟨ …
        ⊢ Eq (Eq.recOn ⋯ ⟨x₁, x₂⟩.snd) ⟨x₁, y₂⟩.snd
      -/
      rw [LinearIndependent.injective (hindep _) hxy]
      /-
        🎉 no goals
      -/
    · have h0 : f x₁ x₂ = 0 := by
        apply
          disjoint_def.1 (hd x₁ {y₁} (finite_singleton y₁) fun h => h_cases (eq_of_mem_singleton h))
            (f x₁ x₂) (subset_span (mem_range_self _))
        rw [iSup_singleton]
        simp only at hxy
        rw [hxy]
        exact subset_span (mem_range_self y₂)
      /-
        case neg
        R : Type u_2
        M : Type u_4
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        η : Type u_6
        ιs : η → Type u_7
        f : (j : η) → ιs j → M
        hindep : ∀ (j : η), LinearIndependent R (f j)
        hd : ∀ (i : η) (t : Set η), t.Finite → Not (Membership.mem t i) → Disjoint (Su …
        a✝ : Nontrivial R
        x₁ : η
        x₂ : ιs x₁
        y₁ : η
        y₂ : ιs y₁
        hxy : Eq ((fun ji => f ji.fst ji.snd) ⟨x₁, x₂⟩) ((fun ji => f ji.fst ji.snd) ⟨ …
        h_cases : Not (Eq x₁ y₁)
        h0 : Eq (f x₁ x₂) 0
        ⊢ Eq ⟨x₁, x₂⟩ ⟨y₁, y₂⟩
      -/
      exact False.elim ((hindep x₁).ne_zero _ h0)
      /-
        🎉 no goals
      -/
  /-
    case a
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    η : Type u_6
    ιs : η → Type u_7
    f : (j : η) → ιs j → M
    hindep : ∀ (j : η), LinearIndependent R (f j)
    hd : ∀ (i : η) (t : Set η), t.Finite → Not (Membership.mem t i) → Disjoint (Su …
    a✝ : Nontrivial R
    ⊢ LinearIndependent R Subtype.val
  -/
  rw [range_sigma_eq_iUnion_range]
  /-
    case a
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    η : Type u_6
    ιs : η → Type u_7
    f : (j : η) → ιs j → M
    hindep : ∀ (j : η), LinearIndependent R (f j)
    hd : ∀ (i : η) (t : Set η), t.Finite → Not (Membership.mem t i) → Disjoint (Su …
    a✝ : Nontrivial R
    ⊢ LinearIndependent R Subtype.val
  -/
  apply linearIndependent_iUnion_finite_subtype (fun j => (hindep j).to_subtype_range) hd
  /-
    🎉 no goals
  -/


/-- Canonical isomorphism between linear combinations and the span of linearly independent vectors.
-/
@[simps (config := { rhsMd := default }) symm_apply]
def LinearIndependent.linearCombinationEquiv (hv : LinearIndependent R v) :
    (ι →₀ R) ≃ₗ[R] span R (range v) := by
  apply LinearEquiv.ofBijective (LinearMap.codRestrict (span R (range v))
                                 (Finsupp.linearCombination R v) _)
  /-
    ι : Type u'
    ι' : Type u_1
    R : Type u_2
    K : Type u_3
    M : Type u_4
    M' : Type u_5
    V : Type u
    v : ι → M
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M
    inst✝ : Module R M'
    hv✝ hv : LinearIndependent R v
    ⊢ Function.Bijective ⇑(LinearMap.codRestrict (Submodule.span R (Set.range v))  …
  -/
  constructor
    /-
      case left
      ι : Type u'
      ι' : Type u_1
      R : Type u_2
      K : Type u_3
      M : Type u_4
      M' : Type u_5
      V : Type u
      v : ι → M
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup M'
      inst✝¹ : Module R M
      inst✝ : Module R M'
      hv✝ hv : LinearIndependent R v
      ⊢ Function.Injective ⇑(LinearMap.codRestrict (Submodule.span R (Set.range v))  …
    -/
  · rw [← LinearMap.ker_eq_bot, LinearMap.ker_codRestrict, ← linearIndependent_iff_ker]
      /-
        case left
        ι : Type u'
        ι' : Type u_1
        R : Type u_2
        K : Type u_3
        M : Type u_4
        M' : Type u_5
        V : Type u
        v : ι → M
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup M'
        inst✝¹ : Module R M
        inst✝ : Module R M'
        hv✝ hv : LinearIndependent R v
        ⊢ LinearIndependent R v
      -/
    · apply hv
      /-
        🎉 no goals
      -/
      /-
        case left.hf
        ι : Type u'
        ι' : Type u_1
        R : Type u_2
        K : Type u_3
        M : Type u_4
        M' : Type u_5
        V : Type u
        v : ι → M
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup M'
        inst✝¹ : Module R M
        inst✝ : Module R M'
        hv✝ hv : LinearIndependent R v
        ⊢ ∀ (c : Finsupp ι R), Membership.mem (Submodule.span R (Set.range v)) ((Finsu …
      -/
    · intro l
      /-
        case left.hf
        ι : Type u'
        ι' : Type u_1
        R : Type u_2
        K : Type u_3
        M : Type u_4
        M' : Type u_5
        V : Type u
        v : ι → M
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup M'
        inst✝¹ : Module R M
        inst✝ : Module R M'
        hv✝ hv : LinearIndependent R v
        l : Finsupp ι R
        ⊢ Membership.mem (Submodule.span R (Set.range v)) ((Finsupp.linearCombination  …
      -/
      rw [← Finsupp.range_linearCombination]
      /-
        case left.hf
        ι : Type u'
        ι' : Type u_1
        R : Type u_2
        K : Type u_3
        M : Type u_4
        M' : Type u_5
        V : Type u
        v : ι → M
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup M'
        inst✝¹ : Module R M
        inst✝ : Module R M'
        hv✝ hv : LinearIndependent R v
        l : Finsupp ι R
        ⊢ Membership.mem (LinearMap.range (Finsupp.linearCombination R v)) ((Finsupp.l …
      -/
      rw [LinearMap.mem_range]
      /-
        case left.hf
        ι : Type u'
        ι' : Type u_1
        R : Type u_2
        K : Type u_3
        M : Type u_4
        M' : Type u_5
        V : Type u
        v : ι → M
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup M'
        inst✝¹ : Module R M
        inst✝ : Module R M'
        hv✝ hv : LinearIndependent R v
        l : Finsupp ι R
        ⊢ Exists fun y => Eq ((Finsupp.linearCombination R v) y) ((Finsupp.linearCombi …
      -/
      apply mem_range_self l
      /-
        🎉 no goals
      -/
  · rw [← LinearMap.range_eq_top, LinearMap.range_eq_map, LinearMap.map_codRestrict,
      ← LinearMap.range_le_iff_comap, range_subtype, Submodule.map_top]
    /-
      case right
      ι : Type u'
      ι' : Type u_1
      R : Type u_2
      K : Type u_3
      M : Type u_4
      M' : Type u_5
      V : Type u
      v : ι → M
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup M'
      inst✝¹ : Module R M
      inst✝ : Module R M'
      hv✝ hv : LinearIndependent R v
      ⊢ LE.le (Submodule.span R (Set.range v)) (LinearMap.range (Finsupp.linearCombi …
    -/
    rw [Finsupp.range_linearCombination]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-08-29")] noncomputable alias LinearIndependent.totalEquiv :=
  LinearIndependent.linearCombinationEquiv

-- Porting note: The original theorem generated by `simps` was
--               different from the theorem on Lean 3, and not simp-normal form.

@[simp]
theorem LinearIndependent.linearCombinationEquiv_apply_coe (hv : LinearIndependent R v)
    (l : ι →₀ R) : hv.linearCombinationEquiv l = Finsupp.linearCombination R v l := rfl


@[deprecated (since := "2024-08-29")] alias LinearIndependent.totalEquiv_apply_coe :=
  LinearIndependent.linearCombinationEquiv_apply_coe

/-- Linear combination representing a vector in the span of linearly independent vectors.

Given a family of linearly independent vectors, we can represent any vector in their span as
a linear combination of these vectors. These are provided by this linear map.
It is simply one direction of `LinearIndependent.linearCombinationEquiv`. -/
def LinearIndependent.repr (hv : LinearIndependent R v) : span R (range v) →ₗ[R] ι →₀ R :=
  hv.linearCombinationEquiv.symm


@[simp]
theorem LinearIndependent.linearCombination_repr (x) :
    Finsupp.linearCombination R v (hv.repr x) = x :=
  Subtype.ext_iff.1 (LinearEquiv.apply_symm_apply hv.linearCombinationEquiv x)


@[deprecated (since := "2024-08-29")] alias LinearIndependent.total_repr :=
  LinearIndependent.linearCombination_repr


theorem LinearIndependent.linearCombination_comp_repr :
    (Finsupp.linearCombination R v).comp hv.repr = Submodule.subtype _ :=
  LinearMap.ext <| hv.linearCombination_repr


@[deprecated (since := "2024-08-29")] alias LinearIndependent.total_comp_repr :=
  LinearIndependent.linearCombination_comp_repr


theorem LinearIndependent.repr_ker : LinearMap.ker hv.repr = ⊥ := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    ⊢ Eq (LinearMap.ker hv.repr) Bot.bot
  -/
  rw [LinearIndependent.repr, LinearEquiv.ker]
  /-
    🎉 no goals
  -/


theorem LinearIndependent.repr_range : LinearMap.range hv.repr = ⊤ := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    ⊢ Eq (LinearMap.range hv.repr) Top.top
  -/
  rw [LinearIndependent.repr, LinearEquiv.range]
  /-
    🎉 no goals
  -/


theorem LinearIndependent.repr_eq {l : ι →₀ R} {x : span R (range v)}
    (eq : Finsupp.linearCombination R v l = ↑x) : hv.repr x = l := by
  have :
    ↑((LinearIndependent.linearCombinationEquiv hv : (ι →₀ R) →ₗ[R] span R (range v)) l) =
      Finsupp.linearCombination R v l :=
    rfl
  have : (LinearIndependent.linearCombinationEquiv hv : (ι →₀ R) →ₗ[R] span R (range v)) l = x := by
    rw [eq] at this
    exact Subtype.ext_iff.2 this
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    l : Finsupp ι R
    x : Subtype fun x => Membership.mem (Submodule.span R (Set.range v)) x
    eq : Eq ((Finsupp.linearCombination R v) l) ↑x
    this✝ : Eq (↑(↑hv.linearCombinationEquiv l)) ((Finsupp.linearCombination R v) l)
    this : Eq (↑hv.linearCombinationEquiv l) x
    ⊢ Eq (hv.repr x) l
  -/
  rw [← LinearEquiv.symm_apply_apply hv.linearCombinationEquiv l]
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    l : Finsupp ι R
    x : Subtype fun x => Membership.mem (Submodule.span R (Set.range v)) x
    eq : Eq ((Finsupp.linearCombination R v) l) ↑x
    this✝ : Eq (↑(↑hv.linearCombinationEquiv l)) ((Finsupp.linearCombination R v) l)
    this : Eq (↑hv.linearCombinationEquiv l) x
    ⊢ Eq (hv.repr x) (hv.linearCombinationEquiv.symm (hv.linearCombinationEquiv l))
  -/
  rw [← this]
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    l : Finsupp ι R
    x : Subtype fun x => Membership.mem (Submodule.span R (Set.range v)) x
    eq : Eq ((Finsupp.linearCombination R v) l) ↑x
    this✝ : Eq (↑(↑hv.linearCombinationEquiv l)) ((Finsupp.linearCombination R v) l)
    this : Eq (↑hv.linearCombinationEquiv l) x
    ⊢ Eq (hv.repr (↑hv.linearCombinationEquiv l)) (hv.linearCombinationEquiv.symm  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem LinearIndependent.repr_eq_single (i) (x : span R (range v)) (hx : ↑x = v i) :
    hv.repr x = Finsupp.single i 1 := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    i : ι
    x : Subtype fun x => Membership.mem (Submodule.span R (Set.range v)) x
    hx : Eq (↑x) (v i)
    ⊢ Eq (hv.repr x) (Finsupp.single i 1)
  -/
  apply hv.repr_eq
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    i : ι
    x : Subtype fun x => Membership.mem (Submodule.span R (Set.range v)) x
    hx : Eq (↑x) (v i)
    ⊢ Eq ((Finsupp.linearCombination R v) (Finsupp.single i 1)) ↑x
  -/
  simp [Finsupp.linearCombination_single, hx]
  /-
    🎉 no goals
  -/


theorem LinearIndependent.span_repr_eq [Nontrivial R] (x) :
    Span.repr R (Set.range v) x =
      (hv.repr x).equivMapDomain (Equiv.ofInjective _ hv.injective) := by
  have p :
    (Span.repr R (Set.range v) x).equivMapDomain (Equiv.ofInjective _ hv.injective).symm =
      hv.repr x := by
    apply (LinearIndependent.linearCombinationEquiv hv).injective
    ext
    simp only [LinearIndependent.linearCombinationEquiv_apply_coe, Equiv.self_comp_ofInjective_symm,
      LinearIndependent.linearCombination_repr, Finsupp.linearCombination_equivMapDomain,
      Span.finsupp_linearCombination_repr]
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    hv : LinearIndependent R v
    inst✝ : Nontrivial R
    x : Subtype fun x => Membership.mem (Submodule.span R (Set.range v)) x
    p : Eq (Finsupp.equivMapDomain (Equiv.ofInjective v ⋯).symm (Span.repr R (Set. …
    ⊢ Eq (Span.repr R (Set.range v) x) (Finsupp.equivMapDomain (Equiv.ofInjective  …
  -/
  ext ⟨_, ⟨i, rfl⟩⟩
  /-
    case h.mk.intro
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    hv : LinearIndependent R v
    inst✝ : Nontrivial R
    x : Subtype fun x => Membership.mem (Submodule.span R (Set.range v)) x
    p : Eq (Finsupp.equivMapDomain (Equiv.ofInjective v ⋯).symm (Span.repr R (Set. …
    i : ι
    ⊢ Eq ((Span.repr R (Set.range v) x) ⟨v i, ⋯⟩) ((Finsupp.equivMapDomain (Equiv. …
  -/
  simp [← p]
  /-
    🎉 no goals
  -/


theorem linearIndependent_iff_not_smul_mem_span :
    LinearIndependent R v ↔ ∀ (i : ι) (a : R), a • v i ∈ span R (v '' (univ \ {i})) → a = 0 :=
  ⟨fun hv i a ha => by
    /-
      ι : Type u'
      R : Type u_2
      M : Type u_4
      v : ι → M
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      hv : LinearIndependent R v
      i : ι
      a : R
      ha : Membership.mem (Submodule.span R (Set.image v (SDiff.sdiff Set.univ (Sing …
      ⊢ Eq a 0
    -/
    rw [Finsupp.span_image_eq_map_linearCombination, mem_map] at ha
    /-
      ι : Type u'
      R : Type u_2
      M : Type u_4
      v : ι → M
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      hv : LinearIndependent R v
      i : ι
      a : R
      ha : Exists fun y => And (Membership.mem (Finsupp.supported R R (SDiff.sdiff S …
      ⊢ Eq a 0
    -/
    rcases ha with ⟨l, hl, e⟩
    /-
      case intro.intro
      ι : Type u'
      R : Type u_2
      M : Type u_4
      v : ι → M
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      hv : LinearIndependent R v
      i : ι
      a : R
      l : Finsupp ι R
      hl : Membership.mem (Finsupp.supported R R (SDiff.sdiff Set.univ (Singleton.si …
      e : Eq ((Finsupp.linearCombination R v) l) (HSMul.hSMul a (v i))
      ⊢ Eq a 0
    -/
    rw [sub_eq_zero.1 (linearIndependent_iff.1 hv (l - Finsupp.single i a) (by simp [e]))] at hl
    /-
      case intro.intro
      ι : Type u'
      R : Type u_2
      M : Type u_4
      v : ι → M
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      hv : LinearIndependent R v
      i : ι
      a : R
      l : Finsupp ι R
      hl : Membership.mem (Finsupp.supported R R (SDiff.sdiff Set.univ (Singleton.si …
      e : Eq ((Finsupp.linearCombination R v) l) (HSMul.hSMul a (v i))
      ⊢ Eq a 0
    -/
    by_contra hn
    /-
      case intro.intro
      ι : Type u'
      R : Type u_2
      M : Type u_4
      v : ι → M
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      hv : LinearIndependent R v
      i : ι
      a : R
      l : Finsupp ι R
      hl : Membership.mem (Finsupp.supported R R (SDiff.sdiff Set.univ (Singleton.si …
      e : Eq ((Finsupp.linearCombination R v) l) (HSMul.hSMul a (v i))
      hn : Not (Eq a 0)
      ⊢ False
    -/
    exact (not_mem_of_mem_diff (hl <| by simp [hn])) (mem_singleton _), fun H =>
    /-
      🎉 no goals
    -/
    linearIndependent_iff.2 fun l hl => by
      /-
        ι : Type u'
        R : Type u_2
        M : Type u_4
        v : ι → M
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        H : ∀ (i : ι) (a : R), Membership.mem (Submodule.span R (Set.image v (SDiff.sd …
        l : Finsupp ι R
        hl : Eq ((Finsupp.linearCombination R v) l) 0
        ⊢ Eq l 0
      -/
      ext i; simp only [Finsupp.zero_apply]
      /-
        case h
        ι : Type u'
        R : Type u_2
        M : Type u_4
        v : ι → M
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        H : ∀ (i : ι) (a : R), Membership.mem (Submodule.span R (Set.image v (SDiff.sd …
        l : Finsupp ι R
        hl : Eq ((Finsupp.linearCombination R v) l) 0
        i : ι
        ⊢ Eq (l i) 0
      -/
      by_contra hn
      /-
        case h
        ι : Type u'
        R : Type u_2
        M : Type u_4
        v : ι → M
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        H : ∀ (i : ι) (a : R), Membership.mem (Submodule.span R (Set.image v (SDiff.sd …
        l : Finsupp ι R
        hl : Eq ((Finsupp.linearCombination R v) l) 0
        i : ι
        hn : Not (Eq (l i) 0)
        ⊢ False
      -/
      refine hn (H i _ ?_)
      /-
        case h
        ι : Type u'
        R : Type u_2
        M : Type u_4
        v : ι → M
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        H : ∀ (i : ι) (a : R), Membership.mem (Submodule.span R (Set.image v (SDiff.sd …
        l : Finsupp ι R
        hl : Eq ((Finsupp.linearCombination R v) l) 0
        i : ι
        hn : Not (Eq (l i) 0)
        ⊢ Membership.mem (Submodule.span R (Set.image v (SDiff.sdiff Set.univ (Singlet …
      -/
      refine (Finsupp.mem_span_image_iff_linearCombination R).2 ⟨Finsupp.single i (l i) - l, ?_, ?_⟩
        /-
          case h.refine_1
          ι : Type u'
          R : Type u_2
          M : Type u_4
          v : ι → M
          inst✝² : Ring R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          H : ∀ (i : ι) (a : R), Membership.mem (Submodule.span R (Set.image v (SDiff.sd …
          l : Finsupp ι R
          hl : Eq ((Finsupp.linearCombination R v) l) 0
          i : ι
          hn : Not (Eq (l i) 0)
          ⊢ Membership.mem (Finsupp.supported R R (SDiff.sdiff Set.univ (Singleton.singl …
        -/
      · rw [Finsupp.mem_supported']
        /-
          case h.refine_1
          ι : Type u'
          R : Type u_2
          M : Type u_4
          v : ι → M
          inst✝² : Ring R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          H : ∀ (i : ι) (a : R), Membership.mem (Submodule.span R (Set.image v (SDiff.sd …
          l : Finsupp ι R
          hl : Eq ((Finsupp.linearCombination R v) l) 0
          i : ι
          hn : Not (Eq (l i) 0)
          ⊢ ∀ (x : ι), Not (Membership.mem (SDiff.sdiff Set.univ (Singleton.singleton i) …
        -/
        intro j hj
        have hij : j = i :=
          Classical.not_not.1 fun hij : j ≠ i =>
            hj ((mem_diff _).2 ⟨mem_univ _, fun h => hij (eq_of_mem_singleton h)⟩)
        /-
          case h.refine_1
          ι : Type u'
          R : Type u_2
          M : Type u_4
          v : ι → M
          inst✝² : Ring R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          H : ∀ (i : ι) (a : R), Membership.mem (Submodule.span R (Set.image v (SDiff.sd …
          l : Finsupp ι R
          hl : Eq ((Finsupp.linearCombination R v) l) 0
          i : ι
          hn : Not (Eq (l i) 0)
          j : ι
          hj : Not (Membership.mem (SDiff.sdiff Set.univ (Singleton.singleton i)) j)
          hij : Eq j i
          ⊢ Eq ((HSub.hSub (Finsupp.single i (l i)) l) j) 0
        -/
        simp [hij]
        /-
          🎉 no goals
        -/
        /-
          case h.refine_2
          ι : Type u'
          R : Type u_2
          M : Type u_4
          v : ι → M
          inst✝² : Ring R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          H : ∀ (i : ι) (a : R), Membership.mem (Submodule.span R (Set.image v (SDiff.sd …
          l : Finsupp ι R
          hl : Eq ((Finsupp.linearCombination R v) l) 0
          i : ι
          hn : Not (Eq (l i) 0)
          ⊢ Eq ((Finsupp.linearCombination R v) (HSub.hSub (Finsupp.single i (l i)) l))  …
        -/
      · simp [hl]⟩
        /-
          🎉 no goals
        -/


/-- See also `iSupIndep_iff_linearIndependent_of_ne_zero`. -/
theorem LinearIndependent.iSupIndep_span_singleton (hv : LinearIndependent R v) :
    iSupIndep fun i => R ∙ v i := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    ⊢ iSupIndep fun i => Submodule.span R (Singleton.singleton (v i))
  -/
  refine iSupIndep_def.mp fun i => ?_
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    i : ι
    ⊢ Disjoint (Submodule.span R (Singleton.singleton (v i))) (iSup fun j => iSup  …
  -/
  rw [disjoint_iff_inf_le]
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    i : ι
    ⊢ LE.le (Min.min (Submodule.span R (Singleton.singleton (v i))) (iSup fun j => …
  -/
  intro m hm
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    i : ι
    m : M
    hm : Membership.mem (Min.min (Submodule.span R (Singleton.singleton (v i))) (i …
    ⊢ Membership.mem Bot.bot m
  -/
  simp only [mem_inf, mem_span_singleton, iSup_subtype'] at hm
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    i : ι
    m : M
    hm : And (Exists fun a => Eq (HSMul.hSMul a (v i)) m) (Membership.mem (iSup fu …
    ⊢ Membership.mem Bot.bot m
  -/
  rw [← span_range_eq_iSup] at hm
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    i : ι
    m : M
    hm : And (Exists fun a => Eq (HSMul.hSMul a (v i)) m) (Membership.mem (Submodu …
    ⊢ Membership.mem Bot.bot m
  -/
  obtain ⟨⟨r, rfl⟩, hm⟩ := hm
  /-
    case intro.intro
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    i : ι
    r : R
    hm : Membership.mem (Submodule.span R (Set.range fun x => v ↑x)) (HSMul.hSMul  …
    ⊢ Membership.mem Bot.bot (HSMul.hSMul r (v i))
  -/
  suffices r = 0 by simp [this]
  /-
    case intro.intro
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    i : ι
    r : R
    hm : Membership.mem (Submodule.span R (Set.range fun x => v ↑x)) (HSMul.hSMul  …
    ⊢ Eq r 0
  -/
  apply linearIndependent_iff_not_smul_mem_span.mp hv i
  /-
    case intro.intro.a
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    i : ι
    r : R
    hm : Membership.mem (Submodule.span R (Set.range fun x => v ↑x)) (HSMul.hSMul  …
    ⊢ Membership.mem (Submodule.span R (Set.image v (SDiff.sdiff Set.univ (Singlet …
  -/
  convert hm
  /-
    case h.e'_4.h.e'_6
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    i : ι
    r : R
    hm : Membership.mem (Submodule.span R (Set.range fun x => v ↑x)) (HSMul.hSMul  …
    ⊢ Eq (Set.image v (SDiff.sdiff Set.univ (Singleton.singleton i))) (Set.range f …
  -/
  ext
  /-
    case h.e'_4.h.e'_6.h
    ι : Type u'
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : LinearIndependent R v
    i : ι
    r : R
    hm : Membership.mem (Submodule.span R (Set.range fun x => v ↑x)) (HSMul.hSMul  …
    x✝ : M
    ⊢ Iff (Membership.mem (Set.image v (SDiff.sdiff Set.univ (Singleton.singleton  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias LinearIndependent.independent_span_singleton := LinearIndependent.iSupIndep_span_singleton


theorem exists_maximal_independent' (s : ι → M) :
    ∃ I : Set ι,
      (LinearIndependent R fun x : I => s x) ∧
        ∀ J : Set ι, I ⊆ J → (LinearIndependent R fun x : J => s x) → I = J := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : ι → M
    ⊢ Exists fun I => And (LinearIndependent R fun x => s ↑x) (∀ (J : Set ι), HasS …
  -/
  let indep : Set ι → Prop := fun I => LinearIndependent R (s ∘ (↑) : I → M)
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : ι → M
    indep : Set ι → Prop := fun I => LinearIndependent R (Function.comp s Subtype. …
    ⊢ Exists fun I => And (LinearIndependent R fun x => s ↑x) (∀ (J : Set ι), HasS …
  -/
  let X := { I : Set ι // indep I }
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : ι → M
    indep : Set ι → Prop := fun I => LinearIndependent R (Function.comp s Subtype. …
    X : Type (max 0 u') := Subtype fun I => indep I
    ⊢ Exists fun I => And (LinearIndependent R fun x => s ↑x) (∀ (J : Set ι), HasS …
  -/
  let r : X → X → Prop := fun I J => I.1 ⊆ J.1
  have key : ∀ c : Set X, IsChain r c → indep (⋃ (I : X) (_ : I ∈ c), I) := by
    intro c hc
    dsimp [indep]
    rw [linearIndependent_comp_subtype]
    intro f hsupport hsum
    rcases eq_empty_or_nonempty c with (rfl | hn)
    · simpa using hsupport
    haveI : IsRefl X r := ⟨fun _ => Set.Subset.refl _⟩
    obtain ⟨I, _I_mem, hI⟩ : ∃ I ∈ c, (f.support : Set ι) ⊆ I :=
      hc.directedOn.exists_mem_subset_of_finset_subset_biUnion hn hsupport
    exact linearIndependent_comp_subtype.mp I.2 f hI hsum
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : ι → M
    indep : Set ι → Prop := fun I => LinearIndependent R (Function.comp s Subtype. …
    X : Type (max 0 u') := Subtype fun I => indep I
    r : X → X → Prop := fun I J => HasSubset.Subset ↑I ↑J
    key : ∀ (c : Set X), IsChain r c → indep (Set.iUnion fun I => Set.iUnion fun x …
    ⊢ Exists fun I => And (LinearIndependent R fun x => s ↑x) (∀ (J : Set ι), HasS …
  -/
  have trans : Transitive r := fun I J K => Set.Subset.trans
  obtain ⟨⟨I, hli : indep I⟩, hmax : ∀ a, r ⟨I, hli⟩ a → r a ⟨I, hli⟩⟩ :=
    exists_maximal_of_chains_bounded
      (fun c hc => ⟨⟨⋃ I ∈ c, (I : Set ι), key c hc⟩, fun I => Set.subset_biUnion_of_mem⟩) @trans
  /-
    case intro.mk
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : ι → M
    indep : Set ι → Prop := fun I => LinearIndependent R (Function.comp s Subtype. …
    X : Type (max 0 u') := Subtype fun I => indep I
    r : X → X → Prop := fun I J => HasSubset.Subset ↑I ↑J
    key : ∀ (c : Set X), IsChain r c → indep (Set.iUnion fun I => Set.iUnion fun x …
    trans : Transitive r
    I : Set ι
    hli : indep I
    hmax : ∀ (a : X), r ⟨I, hli⟩ a → r a ⟨I, hli⟩
    ⊢ Exists fun I => And (LinearIndependent R fun x => s ↑x) (∀ (J : Set ι), HasS …
  -/
  exact ⟨I, hli, fun J hsub hli => Set.Subset.antisymm hsub (hmax ⟨J, hli⟩ hsub)⟩
  /-
    🎉 no goals
  -/


theorem exists_maximal_independent (s : ι → M) :
    ∃ I : Set ι,
      (LinearIndependent R fun x : I => s x) ∧
        ∀ i ∉ I, ∃ a : R, a ≠ 0 ∧ a • s i ∈ span R (s '' I) := by
  classical
    rcases exists_maximal_independent' R s with ⟨I, hIlinind, hImaximal⟩
    use I, hIlinind
    intro i hi
    specialize hImaximal (I ∪ {i}) (by simp)
    set J := I ∪ {i} with hJ
    have memJ : ∀ {x}, x ∈ J ↔ x = i ∨ x ∈ I := by simp [hJ]
    have hiJ : i ∈ J := by simp [J]
    have h := by
      refine mt hImaximal ?_
      · intro h2
        rw [h2] at hi
        exact absurd hiJ hi
    obtain ⟨f, supp_f, sum_f, f_ne⟩ := linearDependent_comp_subtype.mp h
    have hfi : f i ≠ 0 := by
      contrapose hIlinind
      refine linearDependent_comp_subtype.mpr ⟨f, ?_, sum_f, f_ne⟩
      simp only [Finsupp.mem_supported, hJ] at supp_f ⊢
      rintro x hx
      refine (memJ.mp (supp_f hx)).resolve_left ?_
      rintro rfl
      exact hIlinind (Finsupp.mem_support_iff.mp hx)
    use f i, hfi
    have hfi' : i ∈ f.support := Finsupp.mem_support_iff.mpr hfi
    rw [← Finset.insert_erase hfi', Finset.sum_insert (Finset.not_mem_erase _ _),
      add_eq_zero_iff_eq_neg] at sum_f
    rw [sum_f]
    refine neg_mem (sum_mem fun c hc => smul_mem _ _ (subset_span ⟨c, ?_, rfl⟩))
    exact (memJ.mp (supp_f (Finset.erase_subset _ _ hc))).resolve_left (Finset.ne_of_mem_erase hc)


theorem surjective_of_linearIndependent_of_span [Nontrivial R] (hv : LinearIndependent R v)
    (f : ι' ↪ ι) (hss : range v ⊆ span R (range (v ∘ f))) : Surjective f := by
  /-
    ι : Type u'
    ι' : Type u_1
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    f : Function.Embedding ι' ι
    hss : HasSubset.Subset (Set.range v) ↑(Submodule.span R (Set.range (Function.c …
    ⊢ Function.Surjective ⇑f
  -/
  intro i
  /-
    ι : Type u'
    ι' : Type u_1
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    f : Function.Embedding ι' ι
    hss : HasSubset.Subset (Set.range v) ↑(Submodule.span R (Set.range (Function.c …
    i : ι
    ⊢ Exists fun a => Eq (f a) i
  -/
  let repr : (span R (range (v ∘ f)) : Type _) → ι' →₀ R := (hv.comp f f.injective).repr
  /-
    ι : Type u'
    ι' : Type u_1
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    f : Function.Embedding ι' ι
    hss : HasSubset.Subset (Set.range v) ↑(Submodule.span R (Set.range (Function.c …
    i : ι
    repr : (Subtype fun x => Membership.mem (Submodule.span R (Set.range (Function …
    ⊢ Exists fun a => Eq (f a) i
  -/
  let l := (repr ⟨v i, hss (mem_range_self i)⟩).mapDomain f
  have h_total_l : Finsupp.linearCombination R v l = v i := by
    dsimp only [l]
    rw [Finsupp.linearCombination_mapDomain]
    rw [(hv.comp f f.injective).linearCombination_repr]
    -- Porting note: `rfl` isn't necessary.
  have h_total_eq : Finsupp.linearCombination R v l = Finsupp.linearCombination R v
       (Finsupp.single i 1) := by
    rw [h_total_l, Finsupp.linearCombination_single, one_smul]
  /-
    ι : Type u'
    ι' : Type u_1
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    f : Function.Embedding ι' ι
    hss : HasSubset.Subset (Set.range v) ↑(Submodule.span R (Set.range (Function.c …
    i : ι
    repr : (Subtype fun x => Membership.mem (Submodule.span R (Set.range (Function …
    l : Finsupp ι R := Finsupp.mapDomain (⇑f) (repr ⟨v i, ⋯⟩)
    h_total_l : Eq ((Finsupp.linearCombination R v) l) (v i)
    h_total_eq : Eq ((Finsupp.linearCombination R v) l) ((Finsupp.linearCombinatio …
    ⊢ Exists fun a => Eq (f a) i
  -/
  have l_eq : l = _ := hv h_total_eq
  /-
    ι : Type u'
    ι' : Type u_1
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    f : Function.Embedding ι' ι
    hss : HasSubset.Subset (Set.range v) ↑(Submodule.span R (Set.range (Function.c …
    i : ι
    repr : (Subtype fun x => Membership.mem (Submodule.span R (Set.range (Function …
    l : Finsupp ι R := Finsupp.mapDomain (⇑f) (repr ⟨v i, ⋯⟩)
    h_total_l : Eq ((Finsupp.linearCombination R v) l) (v i)
    h_total_eq : Eq ((Finsupp.linearCombination R v) l) ((Finsupp.linearCombinatio …
    l_eq : Eq l (Finsupp.single i 1)
    ⊢ Exists fun a => Eq (f a) i
  -/
  dsimp only [l] at l_eq
  /-
    ι : Type u'
    ι' : Type u_1
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    f : Function.Embedding ι' ι
    hss : HasSubset.Subset (Set.range v) ↑(Submodule.span R (Set.range (Function.c …
    i : ι
    repr : (Subtype fun x => Membership.mem (Submodule.span R (Set.range (Function …
    l : Finsupp ι R := Finsupp.mapDomain (⇑f) (repr ⟨v i, ⋯⟩)
    h_total_l : Eq ((Finsupp.linearCombination R v) l) (v i)
    h_total_eq : Eq ((Finsupp.linearCombination R v) l) ((Finsupp.linearCombinatio …
    l_eq : Eq (Finsupp.mapDomain (⇑f) (repr ⟨v i, ⋯⟩)) (Finsupp.single i 1)
    ⊢ Exists fun a => Eq (f a) i
  -/
  rw [← Finsupp.embDomain_eq_mapDomain] at l_eq
  rcases Finsupp.single_of_embDomain_single (repr ⟨v i, _⟩) f i (1 : R) zero_ne_one.symm l_eq with
    ⟨i', hi'⟩
  /-
    case intro
    ι : Type u'
    ι' : Type u_1
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    f : Function.Embedding ι' ι
    hss : HasSubset.Subset (Set.range v) ↑(Submodule.span R (Set.range (Function.c …
    i : ι
    repr : (Subtype fun x => Membership.mem (Submodule.span R (Set.range (Function …
    l : Finsupp ι R := Finsupp.mapDomain (⇑f) (repr ⟨v i, ⋯⟩)
    h_total_l : Eq ((Finsupp.linearCombination R v) l) (v i)
    h_total_eq : Eq ((Finsupp.linearCombination R v) l) ((Finsupp.linearCombinatio …
    l_eq : Eq (Finsupp.embDomain f (repr ⟨v i, ⋯⟩)) (Finsupp.single i 1)
    i' : ι'
    hi' : And (Eq (repr ⟨v i, ⋯⟩) (Finsupp.single i' 1)) (Eq (f i') i)
    ⊢ Exists fun a => Eq (f a) i
  -/
  use i'
  /-
    case h
    ι : Type u'
    ι' : Type u_1
    R : Type u_2
    M : Type u_4
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    hv : LinearIndependent R v
    f : Function.Embedding ι' ι
    hss : HasSubset.Subset (Set.range v) ↑(Submodule.span R (Set.range (Function.c …
    i : ι
    repr : (Subtype fun x => Membership.mem (Submodule.span R (Set.range (Function …
    l : Finsupp ι R := Finsupp.mapDomain (⇑f) (repr ⟨v i, ⋯⟩)
    h_total_l : Eq ((Finsupp.linearCombination R v) l) (v i)
    h_total_eq : Eq ((Finsupp.linearCombination R v) l) ((Finsupp.linearCombinatio …
    l_eq : Eq (Finsupp.embDomain f (repr ⟨v i, ⋯⟩)) (Finsupp.single i 1)
    i' : ι'
    hi' : And (Eq (repr ⟨v i, ⋯⟩) (Finsupp.single i' 1)) (Eq (f i') i)
    ⊢ Eq (f i') i
  -/
  exact hi'.2
  /-
    🎉 no goals
  -/


theorem eq_of_linearIndependent_of_span_subtype [Nontrivial R] {s t : Set M}
    (hs : LinearIndependent R (fun x => x : s → M)) (h : t ⊆ s) (hst : s ⊆ span R t) : s = t := by
  let f : t ↪ s :=
    ⟨fun x => ⟨x.1, h x.2⟩, fun a b hab => Subtype.coe_injective (Subtype.mk.inj hab)⟩
  have h_surj : Surjective f := by
    apply surjective_of_linearIndependent_of_span hs f _
    convert hst <;> simp [f, comp_def]
  /-
    R : Type u_2
    M : Type u_4
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    s t : Set M
    hs : LinearIndependent R fun x => ↑x
    h : HasSubset.Subset t s
    hst : HasSubset.Subset s ↑(Submodule.span R t)
    f : Function.Embedding ↑t ↑s := { toFun := fun x => ⟨↑x, ⋯⟩, inj' := ⋯ }
    h_surj : Function.Surjective ⇑f
    ⊢ Eq s t
  -/
  show s = t
  /-
    R : Type u_2
    M : Type u_4
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    s t : Set M
    hs : LinearIndependent R fun x => ↑x
    h : HasSubset.Subset t s
    hst : HasSubset.Subset s ↑(Submodule.span R t)
    f : Function.Embedding ↑t ↑s := { toFun := fun x => ⟨↑x, ⋯⟩, inj' := ⋯ }
    h_surj : Function.Surjective ⇑f
    ⊢ Eq s t
  -/
  apply Subset.antisymm _ h
  /-
    R : Type u_2
    M : Type u_4
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    s t : Set M
    hs : LinearIndependent R fun x => ↑x
    h : HasSubset.Subset t s
    hst : HasSubset.Subset s ↑(Submodule.span R t)
    f : Function.Embedding ↑t ↑s := { toFun := fun x => ⟨↑x, ⋯⟩, inj' := ⋯ }
    h_surj : Function.Surjective ⇑f
    ⊢ HasSubset.Subset s t
  -/
  intro x hx
  /-
    R : Type u_2
    M : Type u_4
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    s t : Set M
    hs : LinearIndependent R fun x => ↑x
    h : HasSubset.Subset t s
    hst : HasSubset.Subset s ↑(Submodule.span R t)
    f : Function.Embedding ↑t ↑s := { toFun := fun x => ⟨↑x, ⋯⟩, inj' := ⋯ }
    h_surj : Function.Surjective ⇑f
    x : M
    hx : Membership.mem s x
    ⊢ Membership.mem t x
  -/
  rcases h_surj ⟨x, hx⟩ with ⟨y, hy⟩
  /-
    case intro
    R : Type u_2
    M : Type u_4
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    s t : Set M
    hs : LinearIndependent R fun x => ↑x
    h : HasSubset.Subset t s
    hst : HasSubset.Subset s ↑(Submodule.span R t)
    f : Function.Embedding ↑t ↑s := { toFun := fun x => ⟨↑x, ⋯⟩, inj' := ⋯ }
    h_surj : Function.Surjective ⇑f
    x : M
    hx : Membership.mem s x
    y : ↑t
    hy : Eq (f y) ⟨x, hx⟩
    ⊢ Membership.mem t x
  -/
  convert y.mem
  /-
    case h.e'_5
    R : Type u_2
    M : Type u_4
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    s t : Set M
    hs : LinearIndependent R fun x => ↑x
    h : HasSubset.Subset t s
    hst : HasSubset.Subset s ↑(Submodule.span R t)
    f : Function.Embedding ↑t ↑s := { toFun := fun x => ⟨↑x, ⋯⟩, inj' := ⋯ }
    h_surj : Function.Surjective ⇑f
    x : M
    hx : Membership.mem s x
    y : ↑t
    hy : Eq (f y) ⟨x, hx⟩
    ⊢ Eq x ↑y
  -/
  rw [← Subtype.mk.inj hy]
  /-
    🎉 no goals
  -/


theorem LinearIndependent.image_subtype {s : Set M} {f : M →ₗ[R] M'}
    (hs : LinearIndependent R (fun x => x : s → M))
    (hf_inj : Disjoint (span R s) (LinearMap.ker f)) :
    LinearIndependent R (fun x => x : f '' s → M') := by
  /-
    R : Type u_2
    M : Type u_4
    M' : Type u_5
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M
    inst✝ : Module R M'
    s : Set M
    f : LinearMap (RingHom.id R) M M'
    hs : LinearIndependent R fun x => ↑x
    hf_inj : Disjoint (Submodule.span R s) (LinearMap.ker f)
    ⊢ LinearIndependent R fun x => ↑x
  -/
  rw [← Subtype.range_coe (s := s)] at hf_inj
  /-
    R : Type u_2
    M : Type u_4
    M' : Type u_5
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M
    inst✝ : Module R M'
    s : Set M
    f : LinearMap (RingHom.id R) M M'
    hs : LinearIndependent R fun x => ↑x
    hf_inj : Disjoint (Submodule.span R (Set.range Subtype.val)) (LinearMap.ker f)
    ⊢ LinearIndependent R fun x => ↑x
  -/
  refine (hs.map hf_inj).to_subtype_range' ?_
  /-
    R : Type u_2
    M : Type u_4
    M' : Type u_5
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M
    inst✝ : Module R M'
    s : Set M
    f : LinearMap (RingHom.id R) M M'
    hs : LinearIndependent R fun x => ↑x
    hf_inj : Disjoint (Submodule.span R (Set.range Subtype.val)) (LinearMap.ker f)
    ⊢ Eq (Set.range (Function.comp ⇑f fun x => ↑x)) (Set.image (⇑f) s)
  -/
  simp [Set.range_comp f]
  /-
    🎉 no goals
  -/


theorem LinearIndependent.inl_union_inr {s : Set M} {t : Set M'}
    (hs : LinearIndependent R (fun x => x : s → M))
    (ht : LinearIndependent R (fun x => x : t → M')) :
    LinearIndependent R (fun x => x : ↥(inl R M M' '' s ∪ inr R M M' '' t) → M × M') := by
  /-
    R : Type u_2
    M : Type u_4
    M' : Type u_5
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M
    inst✝ : Module R M'
    s : Set M
    t : Set M'
    hs : LinearIndependent R fun x => ↑x
    ht : LinearIndependent R fun x => ↑x
    ⊢ LinearIndependent R fun x => ↑x
  -/
  refine (hs.image_subtype ?_).union (ht.image_subtype ?_) ?_ <;> [simp; simp; skip]
  -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to change `span_image` into `span_image _`
  /-
    case refine_3
    R : Type u_2
    M : Type u_4
    M' : Type u_5
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M
    inst✝ : Module R M'
    s : Set M
    t : Set M'
    hs : LinearIndependent R fun x => ↑x
    ht : LinearIndependent R fun x => ↑x
    ⊢ Disjoint (Submodule.span R (Set.image (⇑(LinearMap.inl R M M')) s)) (Submodu …
  -/
  simp only [span_image _]
  /-
    case refine_3
    R : Type u_2
    M : Type u_4
    M' : Type u_5
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M
    inst✝ : Module R M'
    s : Set M
    t : Set M'
    hs : LinearIndependent R fun x => ↑x
    ht : LinearIndependent R fun x => ↑x
    ⊢ Disjoint (Submodule.map (LinearMap.inl R M M') (Submodule.span R s)) (Submod …
  -/
  simp [disjoint_iff, prod_inf_prod]
  /-
    🎉 no goals
  -/


theorem linearIndependent_inl_union_inr' {v : ι → M} {v' : ι' → M'} (hv : LinearIndependent R v)
    (hv' : LinearIndependent R v') :
    LinearIndependent R (Sum.elim (inl R M M' ∘ v) (inr R M M' ∘ v')) :=
  (hv.map' (inl R M M') ker_inl).sum_type (hv'.map' (inr R M M') ker_inr) <| by
    /-
      ι : Type u'
      ι' : Type u_1
      R : Type u_2
      M : Type u_4
      M' : Type u_5
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup M'
      inst✝¹ : Module R M
      inst✝ : Module R M'
      v : ι → M
      v' : ι' → M'
      hv : LinearIndependent R v
      hv' : LinearIndependent R v'
      ⊢ Disjoint (Submodule.span R (Set.range (Function.comp (⇑(LinearMap.inl R M M' …
    -/
    refine isCompl_range_inl_inr.disjoint.mono ?_ ?_ <;>
      /-
        case refine_1
        ι : Type u'
        ι' : Type u_1
        R : Type u_2
        M : Type u_4
        M' : Type u_5
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup M'
        inst✝¹ : Module R M
        inst✝ : Module R M'
        v : ι → M
        v' : ι' → M'
        hv : LinearIndependent R v
        hv' : LinearIndependent R v'
        ⊢ LE.le (Submodule.span R (Set.range (Function.comp (⇑(LinearMap.inl R M M'))  …
      -/
      /-
        🎉 no goals
      -/
      simp only [span_le, range_coe, range_comp_subset_range]
      /-
        🎉 no goals
      -/

-- See, for example, Keith Conrad's note
--  <https://kconrad.math.uconn.edu/blurbs/galoistheory/linearchar.pdf>

/-- Dedekind's linear independence of characters -/
@[stacks 0CKL]
theorem linearIndependent_monoidHom (G : Type*) [Monoid G] (L : Type*) [CommRing L]
    [NoZeroDivisors L] : LinearIndependent L (M := G → L) (fun f => f : (G →* L) → G → L) := by
  -- Porting note: Some casts are required.
  /-
    G : Type u_6
    inst✝² : Monoid G
    L : Type u_7
    inst✝¹ : CommRing L
    inst✝ : NoZeroDivisors L
    ⊢ LinearIndependent L fun f => ⇑f
  -/
  letI := Classical.decEq (G →* L)
  /-
    G : Type u_6
    inst✝² : Monoid G
    L : Type u_7
    inst✝¹ : CommRing L
    inst✝ : NoZeroDivisors L
    this : DecidableEq (MonoidHom G L) := Classical.decEq (MonoidHom G L)
    ⊢ LinearIndependent L fun f => ⇑f
  -/
  letI : MulAction L L := DistribMulAction.toMulAction
  -- We prove linear independence by showing that only the trivial linear combination vanishes.
  exact linearIndependent_iff'.2
    -- To do this, we use `Finset` induction,
    -- Porting note: `False.elim` → `fun h => False.elim <| Finset.not_mem_empty _ h`
    fun s =>
      Finset.induction_on s
        (fun g _hg i h => False.elim <| Finset.not_mem_empty _ h) fun a s has ih g hg =>
        -- Here
        -- * `a` is a new character we will insert into the `Finset` of characters `s`,
        -- * `ih` is the fact that only the trivial linear combination of characters in `s` is zero
        -- * `hg` is the fact that `g` are the coefficients of a linear combination summing to zero
        -- and it remains to prove that `g` vanishes on `insert a s`.
        -- We now make the key calculation:
        -- For any character `i` in the original `Finset`, we have `g i • i = g i • a` as functions
        -- on the monoid `G`.
        have h1 : ∀ i ∈ s, (g i • (i : G → L)) = g i • (a : G → L) := fun i his =>
          funext fun x : G =>
            -- We prove these expressions are equal by showing
            -- the differences of their values on each monoid element `x` is zero
            eq_of_sub_eq_zero <|
            ih (fun j => g j * j x - g j * a x)
              (funext fun y : G => calc
                -- After that, it's just a chase scene.
                (∑ i ∈ s, ((g i * i x - g i * a x) • (i : G → L))) y =
                    ∑ i ∈ s, (g i * i x - g i * a x) * i y :=
                  Finset.sum_apply ..
                _ = ∑ i ∈ s, (g i * i x * i y - g i * a x * i y) :=
                  Finset.sum_congr rfl fun _ _ => sub_mul ..
                _ = (∑ i ∈ s, g i * i x * i y) - ∑ i ∈ s, g i * a x * i y :=
                  Finset.sum_sub_distrib
                _ =
                    (g a * a x * a y + ∑ i ∈ s, g i * i x * i y) -
                      (g a * a x * a y + ∑ i ∈ s, g i * a x * i y) := by
                  rw [add_sub_add_left_eq_sub]
                _ =
                    (∑ i ∈ insert a s, g i * i x * i y) -
                      ∑ i ∈ insert a s, g i * a x * i y := by
                  rw [Finset.sum_insert has, Finset.sum_insert has]
                _ =
                    (∑ i ∈ insert a s, g i * i (x * y)) -
                      ∑ i ∈ insert a s, a x * (g i * i y) := by
                  congrm ∑ i ∈ insert a s, ?_ - ∑ i ∈ insert a s, ?_
                  · rw [map_mul, mul_assoc]
                  · rw [mul_assoc, mul_left_comm]
                _ =
                    (∑ i ∈ insert a s, (g i • (i : G → L))) (x * y) -
                      a x * (∑ i ∈ insert a s, (g i • (i : G → L))) y := by
                  rw [Finset.sum_apply, Finset.sum_apply, Finset.mul_sum]; rfl
                _ = 0 - a x * 0 := by rw [hg]; rfl
                _ = 0 := by rw [mul_zero, sub_zero]
                )
              i his
        -- On the other hand, since `a` is not already in `s`, for any character `i ∈ s`
        -- there is some element of the monoid on which it differs from `a`.
        have h2 : ∀ i : G →* L, i ∈ s → ∃ y, i y ≠ a y := fun i his =>
          Classical.by_contradiction fun h =>
            have hia : i = a := MonoidHom.ext fun y =>
              Classical.by_contradiction fun hy => h ⟨y, hy⟩
            has <| hia ▸ his
        -- From these two facts we deduce that `g` actually vanishes on `s`,
        have h3 : ∀ i ∈ s, g i = 0 := fun i his =>
          let ⟨y, hy⟩ := h2 i his
          have h : g i • i y = g i • a y := congr_fun (h1 i his) y
          Or.resolve_right (mul_eq_zero.1 <| by rw [mul_sub, sub_eq_zero]; exact h)
            (sub_ne_zero_of_ne hy)
        -- And so, using the fact that the linear combination over `s` and over `insert a s` both
        -- vanish, we deduce that `g a = 0`.
        have h4 : g a = 0 :=
          calc
            g a = g a * 1 := (mul_one _).symm
            _ = (g a • (a : G → L)) 1 := by rw [← a.map_one]; rfl
            _ = (∑ i ∈ insert a s, (g i • (i : G → L))) 1 := by
              rw [Finset.sum_eq_single a]
              · intro i his hia
                rw [Finset.mem_insert] at his
                rw [h3 i (his.resolve_left hia), zero_smul]
              · intro haas
                exfalso
                apply haas
                exact Finset.mem_insert_self a s
            _ = 0 := by rw [hg]; rfl
        -- Now we're done; the last two facts together imply that `g` vanishes on every element
        -- of `insert a s`.
        (Finset.forall_mem_insert ..).2 ⟨h4, h3⟩


@[stacks 0CKM]
lemma linearIndependent_algHom_toLinearMap
    (K M L) [CommSemiring K] [Semiring M] [Algebra K M] [CommRing L] [IsDomain L] [Algebra K L] :
    LinearIndependent L (AlgHom.toLinearMap : (M →ₐ[K] L) → M →ₗ[K] L) := by
  /-
    K : Type u_6
    M : Type u_7
    L : Type u_8
    inst✝⁵ : CommSemiring K
    inst✝⁴ : Semiring M
    inst✝³ : Algebra K M
    inst✝² : CommRing L
    inst✝¹ : IsDomain L
    inst✝ : Algebra K L
    ⊢ LinearIndependent L AlgHom.toLinearMap
  -/
  apply LinearIndependent.of_comp (LinearMap.ltoFun K M L)
  exact (linearIndependent_monoidHom M L).comp
    (RingHom.toMonoidHom ∘ AlgHom.toRingHom)
    (fun _ _ e ↦ AlgHom.ext (DFunLike.congr_fun e : _))


lemma linearIndependent_algHom_toLinearMap' (K M L) [CommRing K]
    [Semiring M] [Algebra K M] [CommRing L] [IsDomain L] [Algebra K L] [NoZeroSMulDivisors K L] :
    LinearIndependent K (AlgHom.toLinearMap : (M →ₐ[K] L) → M →ₗ[K] L) := by
  /-
    K : Type u_6
    M : Type u_7
    L : Type u_8
    inst✝⁶ : CommRing K
    inst✝⁵ : Semiring M
    inst✝⁴ : Algebra K M
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : NoZeroSMulDivisors K L
    ⊢ LinearIndependent K AlgHom.toLinearMap
  -/
  apply (linearIndependent_algHom_toLinearMap K M L).restrict_scalars
  /-
    K : Type u_6
    M : Type u_7
    L : Type u_8
    inst✝⁶ : CommRing K
    inst✝⁵ : Semiring M
    inst✝⁴ : Algebra K M
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : NoZeroSMulDivisors K L
    ⊢ Function.Injective fun r => HSMul.hSMul r 1
  -/
  simp_rw [Algebra.smul_def, mul_one]
  /-
    K : Type u_6
    M : Type u_7
    L : Type u_8
    inst✝⁶ : CommRing K
    inst✝⁵ : Semiring M
    inst✝⁴ : Algebra K M
    inst✝³ : CommRing L
    inst✝² : IsDomain L
    inst✝¹ : Algebra K L
    inst✝ : NoZeroSMulDivisors K L
    ⊢ Function.Injective fun r => (algebraMap K L) r
  -/
  exact NoZeroSMulDivisors.algebraMap_injective K L
  /-
    🎉 no goals
  -/


theorem le_of_span_le_span [Nontrivial R] {s t u : Set M} (hl : LinearIndependent R ((↑) : u → M))
    (hsu : s ⊆ u) (htu : t ⊆ u) (hst : span R s ≤ span R t) : s ⊆ t := by
  have :=
    eq_of_linearIndependent_of_span_subtype (hl.mono (Set.union_subset hsu htu))
      Set.subset_union_right (Set.union_subset (Set.Subset.trans subset_span hst) subset_span)
  /-
    R : Type u_2
    M : Type u_4
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    s t u : Set M
    hl : LinearIndependent R Subtype.val
    hsu : HasSubset.Subset s u
    htu : HasSubset.Subset t u
    hst : LE.le (Submodule.span R s) (Submodule.span R t)
    this : Eq (Union.union s t) t
    ⊢ HasSubset.Subset s t
  -/
  rw [← this]; apply Set.subset_union_left
               /-
                 🎉 no goals
               -/


theorem span_le_span_iff [Nontrivial R] {s t u : Set M} (hl : LinearIndependent R ((↑) : u → M))
    (hsu : s ⊆ u) (htu : t ⊆ u) : span R s ≤ span R t ↔ s ⊆ t :=
  ⟨le_of_span_le_span hl hsu htu, span_mono⟩


theorem linearIndependent_unique_iff (v : ι → M) [Unique ι] :
    LinearIndependent R v ↔ v default ≠ 0 := by
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝⁵ : Ring R
    inst✝⁴ : Nontrivial R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    v : ι → M
    inst✝ : Unique ι
    ⊢ Iff (LinearIndependent R v) (Ne (v Inhabited.default) 0)
  -/
  simp only [linearIndependent_iff, Finsupp.linearCombination_unique, smul_eq_zero]
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝⁵ : Ring R
    inst✝⁴ : Nontrivial R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    v : ι → M
    inst✝ : Unique ι
    ⊢ Iff (∀ (l : Finsupp ι R), Or (Eq (l Inhabited.default) 0) (Eq (v Inhabited.d …
  -/
  refine ⟨fun h hv => ?_, fun hv l hl => Finsupp.unique_ext <| hl.resolve_right hv⟩
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝⁵ : Ring R
    inst✝⁴ : Nontrivial R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    v : ι → M
    inst✝ : Unique ι
    h : ∀ (l : Finsupp ι R), Or (Eq (l Inhabited.default) 0) (Eq (v Inhabited.defa …
    hv : Eq (v Inhabited.default) 0
    ⊢ False
  -/
  have := h (Finsupp.single default 1) (Or.inr hv)
  /-
    ι : Type u'
    R : Type u_2
    M : Type u_4
    inst✝⁵ : Ring R
    inst✝⁴ : Nontrivial R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    v : ι → M
    inst✝ : Unique ι
    h : ∀ (l : Finsupp ι R), Or (Eq (l Inhabited.default) 0) (Eq (v Inhabited.defa …
    hv : Eq (v Inhabited.default) 0
    this : Eq (Finsupp.single Inhabited.default 1) 0
    ⊢ False
  -/
  exact one_ne_zero (Finsupp.single_eq_zero.1 this)
  /-
    🎉 no goals
  -/


alias ⟨_, linearIndependent_unique⟩ := linearIndependent_unique_iff


theorem linearIndependent_singleton {x : M} (hx : x ≠ 0) :
    LinearIndependent R (fun x => x : ({x} : Set M) → M) :=
  linearIndependent_unique ((↑) : ({x} : Set M) → M) hx


theorem mem_span_insert_exchange :
    x ∈ span K (insert y s) → x ∉ span K s → y ∈ span K (insert x s) := by
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Set V
    x y : V
    ⊢ Membership.mem (Submodule.span K (Insert.insert y s)) x → Not (Membership.me …
  -/
  simp only [mem_span_insert, forall_exists_index, and_imp]
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Set V
    x y : V
    ⊢ ∀ (x_1 : K) (x_2 : V), Membership.mem (Submodule.span K s) x_2 → Eq x (HAdd. …
  -/
  rintro a z hz rfl h
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Set V
    y : V
    a : K
    z : V
    hz : Membership.mem (Submodule.span K s) z
    h : Not (Membership.mem (Submodule.span K s) (HAdd.hAdd (HSMul.hSMul a y) z))
    ⊢ Exists fun a_1 => Exists fun z_1 => And (Membership.mem (Submodule.span K s) …
  -/
  refine ⟨a⁻¹, -a⁻¹ • z, smul_mem _ _ hz, ?_⟩
  have a0 : a ≠ 0 := by
    rintro rfl
    simp_all
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Set V
    y : V
    a : K
    z : V
    hz : Membership.mem (Submodule.span K s) z
    h : Not (Membership.mem (Submodule.span K s) (HAdd.hAdd (HSMul.hSMul a y) z))
    a0 : Ne a 0
    ⊢ Eq y (HAdd.hAdd (HSMul.hSMul (Inv.inv a) (HAdd.hAdd (HSMul.hSMul a y) z)) (H …
  -/
                    /-
                      🎉 no goals
                    -/
  match_scalars <;> simp [a0]
                    /-
                      🎉 no goals
                    -/


theorem linearIndependent_iff_not_mem_span :
    LinearIndependent K v ↔ ∀ i, v i ∉ span K (v '' (univ \ {i})) := by
  /-
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : ι → V
    ⊢ Iff (LinearIndependent K v) (∀ (i : ι), Not (Membership.mem (Submodule.span  …
  -/
  apply linearIndependent_iff_not_smul_mem_span.trans
  /-
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : ι → V
    ⊢ Iff (∀ (i : ι) (a : K), Membership.mem (Submodule.span K (Set.image v (SDiff …
  -/
  constructor
    /-
      case mp
      ι : Type u'
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : ι → V
      ⊢ (∀ (i : ι) (a : K), Membership.mem (Submodule.span K (Set.image v (SDiff.sdi …
    -/
  · intro h i h_in_span
    /-
      case mp
      ι : Type u'
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : ι → V
      h : ∀ (i : ι) (a : K), Membership.mem (Submodule.span K (Set.image v (SDiff.sd …
      i : ι
      h_in_span : Membership.mem (Submodule.span K (Set.image v (SDiff.sdiff Set.uni …
      ⊢ False
    -/
    apply one_ne_zero (h i 1 (by simp [h_in_span]))
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u'
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : ι → V
      ⊢ (∀ (i : ι), Not (Membership.mem (Submodule.span K (Set.image v (SDiff.sdiff  …
    -/
  · intro h i a ha
    /-
      case mpr
      ι : Type u'
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : ι → V
      h : ∀ (i : ι), Not (Membership.mem (Submodule.span K (Set.image v (SDiff.sdiff …
      i : ι
      a : K
      ha : Membership.mem (Submodule.span K (Set.image v (SDiff.sdiff Set.univ (Sing …
      ⊢ Eq a 0
    -/
    by_contra ha'
    /-
      case mpr
      ι : Type u'
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : ι → V
      h : ∀ (i : ι), Not (Membership.mem (Submodule.span K (Set.image v (SDiff.sdiff …
      i : ι
      a : K
      ha : Membership.mem (Submodule.span K (Set.image v (SDiff.sdiff Set.univ (Sing …
      ha' : Not (Eq a 0)
      ⊢ False
    -/
    exact False.elim (h _ ((smul_mem_iff _ ha').1 ha))
    /-
      🎉 no goals
    -/


protected theorem LinearIndependent.insert (hs : LinearIndependent K (fun b => b : s → V))
    (hx : x ∉ span K s) : LinearIndependent K (fun b => b : ↥(insert x s) → V) := by
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Set V
    x : V
    hs : LinearIndependent K fun b => ↑b
    hx : Not (Membership.mem (Submodule.span K s) x)
    ⊢ LinearIndependent K fun b => ↑b
  -/
  rw [← union_singleton]
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Set V
    x : V
    hs : LinearIndependent K fun b => ↑b
    hx : Not (Membership.mem (Submodule.span K s) x)
    ⊢ LinearIndependent K fun b => ↑b
  -/
  have x0 : x ≠ 0 := mt (by rintro rfl; apply zero_mem (span K s)) hx
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Set V
    x : V
    hs : LinearIndependent K fun b => ↑b
    hx : Not (Membership.mem (Submodule.span K s) x)
    x0 : Ne x 0
    ⊢ LinearIndependent K fun b => ↑b
  -/
  apply hs.union (linearIndependent_singleton x0)
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Set V
    x : V
    hs : LinearIndependent K fun b => ↑b
    hx : Not (Membership.mem (Submodule.span K s) x)
    x0 : Ne x 0
    ⊢ Disjoint (Submodule.span K s) (Submodule.span K (Singleton.singleton x))
  -/
  rwa [disjoint_span_singleton' x0]
  /-
    🎉 no goals
  -/


theorem linearIndependent_option' :
    LinearIndependent K (fun o => Option.casesOn' o x v : Option ι → V) ↔
      LinearIndependent K v ∧ x ∉ Submodule.span K (range v) := by
  -- Porting note: Explicit universe level is required in `Equiv.optionEquivSumPUnit`.
  rw [← linearIndependent_equiv (Equiv.optionEquivSumPUnit.{u', _} ι).symm, linearIndependent_sum,
    @range_unique _ PUnit, @linearIndependent_unique_iff PUnit, disjoint_span_singleton]
  /-
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : ι → V
    x : V
    ⊢ Iff (And (LinearIndependent K (Function.comp (Function.comp (fun o => o.case …
  -/
  dsimp [(· ∘ ·)]
  /-
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : ι → V
    x : V
    ⊢ Iff (And (LinearIndependent K (Function.comp (Function.comp (fun o => o.case …
  -/
  refine ⟨fun h => ⟨h.1, fun hx => h.2.1 <| h.2.2 hx⟩, fun h => ⟨h.1, ?_, fun hx => (h.2 hx).elim⟩⟩
  /-
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : ι → V
    x : V
    h : And (LinearIndependent K v) (Not (Membership.mem (Submodule.span K (Set.ra …
    ⊢ Not (Eq x 0)
  -/
  rintro rfl
  /-
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : ι → V
    h : And (LinearIndependent K v) (Not (Membership.mem (Submodule.span K (Set.ra …
    ⊢ False
  -/
  exact h.2 (zero_mem _)
  /-
    🎉 no goals
  -/


theorem LinearIndependent.option (hv : LinearIndependent K v)
    (hx : x ∉ Submodule.span K (range v)) :
    LinearIndependent K (fun o => Option.casesOn' o x v : Option ι → V) :=
  linearIndependent_option'.2 ⟨hv, hx⟩


theorem linearIndependent_option {v : Option ι → V} : LinearIndependent K v ↔
    LinearIndependent K (v ∘ (↑) : ι → V) ∧
      v none ∉ Submodule.span K (range (v ∘ (↑) : ι → V)) := by
  /-
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : Option ι → V
    ⊢ Iff (LinearIndependent K v) (And (LinearIndependent K (Function.comp v Optio …
  -/
  simp only [← linearIndependent_option', Option.casesOn'_none_coe]
  /-
    🎉 no goals
  -/


theorem linearIndependent_insert' {ι} {s : Set ι} {a : ι} {f : ι → V} (has : a ∉ s) :
    (LinearIndependent K fun x : ↥(insert a s) => f x) ↔
      (LinearIndependent K fun x : s => f x) ∧ f a ∉ Submodule.span K (f '' s) := by
  classical
  rw [← linearIndependent_equiv ((Equiv.optionEquivSumPUnit _).trans (Equiv.Set.insert has).symm),
    linearIndependent_option]
  -- Porting note: `simp [(· ∘ ·), range_comp f]` → `simp [(· ∘ ·)]; erw [range_comp f ..]; simp`
  -- https://github.com/leanprover-community/mathlib4/issues/5164
  simp only [Function.comp_def]
  erw [range_comp f ((↑) : s → ι)]
  simp


theorem linearIndependent_insert (hxs : x ∉ s) :
    (LinearIndependent K fun b : ↥(insert x s) => (b : V)) ↔
      (LinearIndependent K fun b : s => (b : V)) ∧ x ∉ Submodule.span K s :=
                                                        /-
                                                          K : Type u_3
                                                          V : Type u
                                                          inst✝² : DivisionRing K
                                                          inst✝¹ : AddCommGroup V
                                                          inst✝ : Module K V
                                                          s : Set V
                                                          x : V
                                                          hxs : Not (Membership.mem s x)
                                                          ⊢ Iff (And (LinearIndependent K fun x => id ↑x) (Not (Membership.mem (Submodul …
                                                        -/
  (linearIndependent_insert' (f := id) hxs).trans <| by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem linearIndependent_pair {x y : V} (hx : x ≠ 0) (hy : ∀ a : K, a • x ≠ y) :
    LinearIndependent K ((↑) : ({x, y} : Set V) → V) :=
  pair_comm y x ▸ (linearIndependent_singleton hx).insert <|
    mt mem_span_singleton.1 (not_exists.2 hy)


/-- Also see `LinearIndependent.pair_iff` for the version over arbitrary rings. -/
theorem LinearIndependent.pair_iff' {x y : V} (hx : x ≠ 0) :
    LinearIndependent K ![x, y] ↔ ∀ a : K, a • x ≠ y := by
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x y : V
    hx : Ne x 0
    ⊢ Iff (LinearIndependent K (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty …
  -/
  rw [LinearIndependent.pair_iff]
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x y : V
    hx : Ne x 0
    ⊢ Iff (∀ (s t : K), Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0 → And …
  -/
  constructor
    /-
      case mp
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      x y : V
      hx : Ne x 0
      ⊢ (∀ (s t : K), Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0 → And (Eq …
    -/
  · intro H a ha
    /-
      case mp
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      x y : V
      hx : Ne x 0
      H : ∀ (s t : K), Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0 → And (E …
      a : K
      ha : Eq (HSMul.hSMul a x) y
      ⊢ False
    -/
    have := (H a (-1) (by simpa [← sub_eq_add_neg, sub_eq_zero])).2
    /-
      case mp
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      x y : V
      hx : Ne x 0
      H : ∀ (s t : K), Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0 → And (E …
      a : K
      ha : Eq (HSMul.hSMul a x) y
      this : Eq (-1) 0
      ⊢ False
    -/
    simp only [neg_eq_zero, one_ne_zero] at this
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      x y : V
      hx : Ne x 0
      ⊢ (∀ (a : K), Ne (HSMul.hSMul a x) y) → ∀ (s t : K), Eq (HAdd.hAdd (HSMul.hSMu …
    -/
  · intro H s t hst
    /-
      case mpr
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      x y : V
      hx : Ne x 0
      H : ∀ (a : K), Ne (HSMul.hSMul a x) y
      s t : K
      hst : Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0
      ⊢ And (Eq s 0) (Eq t 0)
    -/
    by_cases ht : t = 0
      /-
        case pos
        K : Type u_3
        V : Type u
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        x y : V
        hx : Ne x 0
        H : ∀ (a : K), Ne (HSMul.hSMul a x) y
        s t : K
        hst : Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0
        ht : Eq t 0
        ⊢ And (Eq s 0) (Eq t 0)
      -/
    · exact ⟨by simpa [ht, hx] using hst, ht⟩
      /-
        🎉 no goals
      -/
    /-
      case neg
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      x y : V
      hx : Ne x 0
      H : ∀ (a : K), Ne (HSMul.hSMul a x) y
      s t : K
      hst : Eq (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t y)) 0
      ht : Not (Eq t 0)
      ⊢ And (Eq s 0) (Eq t 0)
    -/
    apply_fun (t⁻¹ • ·) at hst
    /-
      case neg
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      x y : V
      hx : Ne x 0
      H : ∀ (a : K), Ne (HSMul.hSMul a x) y
      s t : K
      ht : Not (Eq t 0)
      hst : Eq (HSMul.hSMul (Inv.inv t) (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul t  …
      ⊢ And (Eq s 0) (Eq t 0)
    -/
    simp only [smul_add, smul_smul, inv_mul_cancel₀ ht] at hst
    /-
      case neg
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      x y : V
      hx : Ne x 0
      H : ∀ (a : K), Ne (HSMul.hSMul a x) y
      s t : K
      ht : Not (Eq t 0)
      hst : Eq (HAdd.hAdd (HSMul.hSMul (HMul.hMul (Inv.inv t) s) x) (HSMul.hSMul 1 y …
      ⊢ And (Eq s 0) (Eq t 0)
    -/
    cases H (-(t⁻¹ * s)) <| by linear_combination (norm := match_scalars <;> noncomm_ring) -hst
    /-
      🎉 no goals
    -/


theorem linearIndependent_fin_cons {n} {v : Fin n → V} :
    LinearIndependent K (Fin.cons x v : Fin (n + 1) → V) ↔
      LinearIndependent K v ∧ x ∉ Submodule.span K (range v) := by
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x : V
    n : Nat
    v : Fin n → V
    ⊢ Iff (LinearIndependent K (Fin.cons x v)) (And (LinearIndependent K v) (Not ( …
  -/
  rw [← linearIndependent_equiv (finSuccEquiv n).symm, linearIndependent_option]
  -- Porting note: `convert Iff.rfl; ...` → `exact Iff.rfl`
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x : V
    n : Nat
    v : Fin n → V
    ⊢ Iff (And (LinearIndependent K (Function.comp (Function.comp (Fin.cons x v) ⇑ …
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


theorem linearIndependent_fin_snoc {n} {v : Fin n → V} :
    LinearIndependent K (Fin.snoc v x : Fin (n + 1) → V) ↔
      LinearIndependent K v ∧ x ∉ Submodule.span K (range v) := by
  -- Porting note: `rw` → `erw`
  -- https://github.com/leanprover-community/mathlib4/issues/5164
  -- Here Lean can not see that `fun i ↦ Fin.cons x v (↑(finRotate (n + 1)) i)`
  -- matches with `?f ∘ ↑(finRotate (n + 1))`.
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x : V
    n : Nat
    v : Fin n → V
    ⊢ Iff (LinearIndependent K (Fin.snoc v x)) (And (LinearIndependent K v) (Not ( …
  -/
  erw [Fin.snoc_eq_cons_rotate, linearIndependent_equiv, linearIndependent_fin_cons]
  /-
    🎉 no goals
  -/


/-- See `LinearIndependent.fin_cons'` for an uglier version that works if you
only have a module over a semiring. -/
theorem LinearIndependent.fin_cons {n} {v : Fin n → V} (hv : LinearIndependent K v)
    (hx : x ∉ Submodule.span K (range v)) : LinearIndependent K (Fin.cons x v : Fin (n + 1) → V) :=
  linearIndependent_fin_cons.2 ⟨hv, hx⟩


theorem linearIndependent_fin_succ {n} {v : Fin (n + 1) → V} :
    LinearIndependent K v ↔
      LinearIndependent K (Fin.tail v) ∧ v 0 ∉ Submodule.span K (range <| Fin.tail v) := by
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    n : Nat
    v : Fin (HAdd.hAdd n 1) → V
    ⊢ Iff (LinearIndependent K v) (And (LinearIndependent K (Fin.tail v)) (Not (Me …
  -/
  rw [← linearIndependent_fin_cons, Fin.cons_self_tail]
  /-
    🎉 no goals
  -/


theorem linearIndependent_fin_succ' {n} {v : Fin (n + 1) → V} : LinearIndependent K v ↔
    LinearIndependent K (Fin.init v) ∧ v (Fin.last _) ∉ Submodule.span K (range <| Fin.init v) := by
  /-
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    n : Nat
    v : Fin (HAdd.hAdd n 1) → V
    ⊢ Iff (LinearIndependent K v) (And (LinearIndependent K (Fin.init v)) (Not (Me …
  -/
  rw [← linearIndependent_fin_snoc, Fin.snoc_init_self]
  /-
    🎉 no goals
  -/


/-- Equivalence between `k + 1` vectors of length `n` and `k` vectors of length `n` along with a
vector in the complement of their span.
-/
def equiv_linearIndependent (n : ℕ) :
    { s : Fin (n + 1) → V // LinearIndependent K s } ≃
      Σ s : { s : Fin n → V // LinearIndependent K s },
        ((Submodule.span K (Set.range (s : Fin n → V)))ᶜ : Set V) where
  toFun s := ⟨⟨Fin.tail s.val, (linearIndependent_fin_succ.mp s.property).left⟩,
    ⟨s.val 0, (linearIndependent_fin_succ.mp s.property).right⟩⟩
  invFun s := ⟨Fin.cons s.2.val s.1.val,
    linearIndependent_fin_cons.mpr ⟨s.1.property, s.2.property⟩⟩
                   /-
                     ι : Type u'
                     ι' : Type u_1
                     R : Type u_2
                     K : Type u_3
                     M : Type u_4
                     M' : Type u_5
                     V : Type u
                     inst✝² : DivisionRing K
                     inst✝¹ : AddCommGroup V
                     inst✝ : Module K V
                     v : ι → V
                     s t : Set V
                     x y : V
                     n : Nat
                     x✝ : Subtype fun s => LinearIndependent K s
                     ⊢ Eq ((fun s => ⟨Fin.cons ↑s.snd ↑s.fst, ⋯⟩) ((fun s => ⟨⟨Fin.tail ↑s, ⋯⟩, ⟨↑s …
                   -/
  left_inv _ := by simp only [Fin.cons_self_tail, Subtype.coe_eta]
                   /-
                     🎉 no goals
                   -/
  right_inv := fun ⟨_, _⟩ => by simp only [Fin.cons_zero, Subtype.coe_eta, Sigma.mk.inj_iff,
    Fin.tail_cons, heq_eq_eq, and_self]


theorem linearIndependent_fin2 {f : Fin 2 → V} :
    LinearIndependent K f ↔ f 1 ≠ 0 ∧ ∀ a : K, a • f 1 ≠ f 0 := by
  rw [linearIndependent_fin_succ, linearIndependent_unique_iff, range_unique, mem_span_singleton,
    not_exists, show Fin.tail f default = f 1 by rw [← Fin.succ_zero_eq_one]; rfl]


theorem exists_linearIndependent_extension (hs : LinearIndependent K ((↑) : s → V)) (hst : s ⊆ t) :
    ∃ b ⊆ t, s ⊆ b ∧ t ⊆ span K b ∧ LinearIndependent K ((↑) : b → V) := by
  obtain ⟨b, sb, h⟩ := by
    refine zorn_subset_nonempty { b | b ⊆ t ∧ LinearIndependent K ((↑) : b → V) } ?_ _ ⟨hst, hs⟩
    · refine fun c hc cc _c0 => ⟨⋃₀ c, ⟨?_, ?_⟩, fun x => ?_⟩
      · exact sUnion_subset fun x xc => (hc xc).1
      · exact linearIndependent_sUnion_of_directed cc.directedOn fun x xc => (hc xc).2
      · exact subset_sUnion_of_mem
  /-
    case intro.intro
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s t : Set V
    hs : LinearIndependent K Subtype.val
    hst : HasSubset.Subset s t
    b : Set V
    sb : HasSubset.Subset s b
    h : Maximal (fun x => Membership.mem (setOf fun b => And (HasSubset.Subset b t …
    ⊢ Exists fun b => And (HasSubset.Subset b t) (And (HasSubset.Subset s b) (And  …
  -/
  refine ⟨b, h.prop.1, sb, fun x xt => by_contra fun hn ↦ hn ?_, h.prop.2⟩
  /-
    case intro.intro
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s t : Set V
    hs : LinearIndependent K Subtype.val
    hst : HasSubset.Subset s t
    b : Set V
    sb : HasSubset.Subset s b
    h : Maximal (fun x => Membership.mem (setOf fun b => And (HasSubset.Subset b t …
    x : V
    xt : Membership.mem t x
    hn : Not (Membership.mem (↑(Submodule.span K b)) x)
    ⊢ Membership.mem (↑(Submodule.span K b)) x
  -/
  exact subset_span <| h.mem_of_prop_insert ⟨insert_subset xt h.prop.1, h.prop.2.insert hn⟩
  /-
    🎉 no goals
  -/


theorem exists_linearIndependent :
    ∃ b ⊆ t, span K b = span K t ∧ LinearIndependent K ((↑) : b → V) := by
  obtain ⟨b, hb₁, -, hb₂, hb₃⟩ :=
    exists_linearIndependent_extension (linearIndependent_empty K V) (Set.empty_subset t)
  /-
    case intro.intro.intro.intro
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    t b : Set V
    hb₁ : HasSubset.Subset b t
    hb₂ : HasSubset.Subset t ↑(Submodule.span K b)
    hb₃ : LinearIndependent K Subtype.val
    ⊢ Exists fun b => And (HasSubset.Subset b t) (And (Eq (Submodule.span K b) (Su …
  -/
  exact ⟨b, hb₁, (span_eq_of_le _ hb₂ (Submodule.span_mono hb₁)).symm, hb₃⟩
  /-
    🎉 no goals
  -/


/-- Indexed version of `exists_linearIndependent`. -/
lemma exists_linearIndependent' (v : ι → V) :
    ∃ (κ : Type u') (a : κ → ι), Function.Injective a ∧
      Submodule.span K (Set.range (v ∘ a)) = Submodule.span K (Set.range v) ∧
      LinearIndependent K (v ∘ a) := by
  /-
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : ι → V
    ⊢ Exists fun κ => Exists fun a => And (Function.Injective a) (And (Eq (Submodu …
  -/
  obtain ⟨t, ht, hsp, hli⟩ := exists_linearIndependent K (Set.range v)
  /-
    case intro.intro.intro
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : ι → V
    t : Set V
    ht : HasSubset.Subset t (Set.range v)
    hsp : Eq (Submodule.span K t) (Submodule.span K (Set.range v))
    hli : LinearIndependent K Subtype.val
    ⊢ Exists fun κ => Exists fun a => And (Function.Injective a) (And (Eq (Submodu …
  -/
  choose f hf using ht
  /-
    case intro.intro.intro
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : ι → V
    t : Set V
    hsp : Eq (Submodule.span K t) (Submodule.span K (Set.range v))
    hli : LinearIndependent K Subtype.val
    f : ⦃a : V⦄ → Membership.mem t a → ι
    hf : ∀ ⦃a : V⦄ (a_1 : Membership.mem t a), Eq (v (f a_1)) a
    ⊢ Exists fun κ => Exists fun a => And (Function.Injective a) (And (Eq (Submodu …
  -/
  let s : Set ι := Set.range (fun a : t ↦ f a.property)
  /-
    case intro.intro.intro
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : ι → V
    t : Set V
    hsp : Eq (Submodule.span K t) (Submodule.span K (Set.range v))
    hli : LinearIndependent K Subtype.val
    f : ⦃a : V⦄ → Membership.mem t a → ι
    hf : ∀ ⦃a : V⦄ (a_1 : Membership.mem t a), Eq (v (f a_1)) a
    s : Set ι := Set.range fun a => f ⋯
    ⊢ Exists fun κ => Exists fun a => And (Function.Injective a) (And (Eq (Submodu …
  -/
  have hs {i : ι} (hi : i ∈ s) : v i ∈ t := by obtain ⟨a, rfl⟩ := hi; simp [hf]
  /-
    case intro.intro.intro
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : ι → V
    t : Set V
    hsp : Eq (Submodule.span K t) (Submodule.span K (Set.range v))
    hli : LinearIndependent K Subtype.val
    f : ⦃a : V⦄ → Membership.mem t a → ι
    hf : ∀ ⦃a : V⦄ (a_1 : Membership.mem t a), Eq (v (f a_1)) a
    s : Set ι := Set.range fun a => f ⋯
    hs : ∀ {i : ι}, Membership.mem s i → Membership.mem t (v i)
    ⊢ Exists fun κ => Exists fun a => And (Function.Injective a) (And (Eq (Submodu …
  -/
  let f' (a : s) : t := ⟨v a.val, hs a.property⟩
  /-
    case intro.intro.intro
    ι : Type u'
    K : Type u_3
    V : Type u
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : ι → V
    t : Set V
    hsp : Eq (Submodule.span K t) (Submodule.span K (Set.range v))
    hli : LinearIndependent K Subtype.val
    f : ⦃a : V⦄ → Membership.mem t a → ι
    hf : ∀ ⦃a : V⦄ (a_1 : Membership.mem t a), Eq (v (f a_1)) a
    s : Set ι := Set.range fun a => f ⋯
    hs : ∀ {i : ι}, Membership.mem s i → Membership.mem t (v i)
    f' : ↑s → ↑t := fun a => ⟨v ↑a, ⋯⟩
    ⊢ Exists fun κ => Exists fun a => And (Function.Injective a) (And (Eq (Submodu …
  -/
  refine ⟨s, Subtype.val, Subtype.val_injective, hsp.symm ▸ by congr; aesop, ?_⟩
    /-
      case intro.intro.intro
      ι : Type u'
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : ι → V
      t : Set V
      hsp : Eq (Submodule.span K t) (Submodule.span K (Set.range v))
      hli : LinearIndependent K Subtype.val
      f : ⦃a : V⦄ → Membership.mem t a → ι
      hf : ∀ ⦃a : V⦄ (a_1 : Membership.mem t a), Eq (v (f a_1)) a
      s : Set ι := Set.range fun a => f ⋯
      hs : ∀ {i : ι}, Membership.mem s i → Membership.mem t (v i)
      f' : ↑s → ↑t := fun a => ⟨v ↑a, ⋯⟩
      ⊢ LinearIndependent K (Function.comp v Subtype.val)
    -/
  · rw [← show Subtype.val ∘ f' = v ∘ Subtype.val by ext; simp]
    /-
      case intro.intro.intro
      ι : Type u'
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : ι → V
      t : Set V
      hsp : Eq (Submodule.span K t) (Submodule.span K (Set.range v))
      hli : LinearIndependent K Subtype.val
      f : ⦃a : V⦄ → Membership.mem t a → ι
      hf : ∀ ⦃a : V⦄ (a_1 : Membership.mem t a), Eq (v (f a_1)) a
      s : Set ι := Set.range fun a => f ⋯
      hs : ∀ {i : ι}, Membership.mem s i → Membership.mem t (v i)
      f' : ↑s → ↑t := fun a => ⟨v ↑a, ⋯⟩
      ⊢ LinearIndependent K (Function.comp Subtype.val f')
    -/
    apply hli.comp
    /-
      case intro.intro.intro.hf
      ι : Type u'
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : ι → V
      t : Set V
      hsp : Eq (Submodule.span K t) (Submodule.span K (Set.range v))
      hli : LinearIndependent K Subtype.val
      f : ⦃a : V⦄ → Membership.mem t a → ι
      hf : ∀ ⦃a : V⦄ (a_1 : Membership.mem t a), Eq (v (f a_1)) a
      s : Set ι := Set.range fun a => f ⋯
      hs : ∀ {i : ι}, Membership.mem s i → Membership.mem t (v i)
      f' : ↑s → ↑t := fun a => ⟨v ↑a, ⋯⟩
      ⊢ Function.Injective f'
    -/
    rintro ⟨i, x, rfl⟩ ⟨j, y, rfl⟩ hij
    /-
      case intro.intro.intro.hf.mk.intro.mk.intro
      ι : Type u'
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : ι → V
      t : Set V
      hsp : Eq (Submodule.span K t) (Submodule.span K (Set.range v))
      hli : LinearIndependent K Subtype.val
      f : ⦃a : V⦄ → Membership.mem t a → ι
      hf : ∀ ⦃a : V⦄ (a_1 : Membership.mem t a), Eq (v (f a_1)) a
      s : Set ι := Set.range fun a => f ⋯
      hs : ∀ {i : ι}, Membership.mem s i → Membership.mem t (v i)
      f' : ↑s → ↑t := fun a => ⟨v ↑a, ⋯⟩
      x y : ↑t
      hij : Eq (f' ⟨(fun a => f ⋯) x, ⋯⟩) (f' ⟨(fun a => f ⋯) y, ⋯⟩)
      ⊢ Eq ⟨(fun a => f ⋯) x, ⋯⟩ ⟨(fun a => f ⋯) y, ⋯⟩
    -/
    simp only [Subtype.ext_iff, hf] at hij
    /-
      case intro.intro.intro.hf.mk.intro.mk.intro
      ι : Type u'
      K : Type u_3
      V : Type u
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : ι → V
      t : Set V
      hsp : Eq (Submodule.span K t) (Submodule.span K (Set.range v))
      hli : LinearIndependent K Subtype.val
      f : ⦃a : V⦄ → Membership.mem t a → ι
      hf : ∀ ⦃a : V⦄ (a_1 : Membership.mem t a), Eq (v (f a_1)) a
      s : Set ι := Set.range fun a => f ⋯
      hs : ∀ {i : ι}, Membership.mem s i → Membership.mem t (v i)
      f' : ↑s → ↑t := fun a => ⟨v ↑a, ⋯⟩
      x y : ↑t
      hij : Eq ↑x ↑y
      ⊢ Eq ⟨(fun a => f ⋯) x, ⋯⟩ ⟨(fun a => f ⋯) y, ⋯⟩
    -/
    simp [hij]
    /-
      🎉 no goals
    -/


/-- `LinearIndependent.extend` adds vectors to a linear independent set `s ⊆ t` until it spans
all elements of `t`. -/
noncomputable def LinearIndependent.extend (hs : LinearIndependent K (fun x => x : s → V))
    (hst : s ⊆ t) : Set V :=
  Classical.choose (exists_linearIndependent_extension hs hst)


theorem LinearIndependent.extend_subset (hs : LinearIndependent K (fun x => x : s → V))
    (hst : s ⊆ t) : hs.extend hst ⊆ t :=
  let ⟨hbt, _hsb, _htb, _hli⟩ := Classical.choose_spec (exists_linearIndependent_extension hs hst)
  hbt


theorem LinearIndependent.subset_extend (hs : LinearIndependent K (fun x => x : s → V))
    (hst : s ⊆ t) : s ⊆ hs.extend hst :=
  let ⟨_hbt, hsb, _htb, _hli⟩ := Classical.choose_spec (exists_linearIndependent_extension hs hst)
  hsb


theorem LinearIndependent.subset_span_extend (hs : LinearIndependent K (fun x => x : s → V))
    (hst : s ⊆ t) : t ⊆ span K (hs.extend hst) :=
  let ⟨_hbt, _hsb, htb, _hli⟩ := Classical.choose_spec (exists_linearIndependent_extension hs hst)
  htb


theorem LinearIndependent.span_extend_eq_span (hs : LinearIndependent K (fun x => x : s → V))
    (hst : s ⊆ t) : span K (hs.extend hst) = span K t :=
  le_antisymm (span_mono (hs.extend_subset hst)) (span_le.2 (hs.subset_span_extend hst))


theorem LinearIndependent.linearIndependent_extend (hs : LinearIndependent K (fun x => x : s → V))
    (hst : s ⊆ t) : LinearIndependent K ((↑) : hs.extend hst → V) :=
  let ⟨_hbt, _hsb, _htb, hli⟩ := Classical.choose_spec (exists_linearIndependent_extension hs hst)
  hli

-- TODO(Mario): rewrite?

theorem exists_of_linearIndependent_of_finite_span {t : Finset V}
    (hs : LinearIndependent K (fun x => x : s → V)) (hst : s ⊆ (span K ↑t : Submodule K V)) :
    ∃ t' : Finset V, ↑t' ⊆ s ∪ ↑t ∧ s ⊆ ↑t' ∧ t'.card = t.card := by
  classical
  have :
    ∀ t : Finset V,
      ∀ s' : Finset V,
        ↑s' ⊆ s →
          s ∩ ↑t = ∅ →
            s ⊆ (span K ↑(s' ∪ t) : Submodule K V) →
              ∃ t' : Finset V, ↑t' ⊆ s ∪ ↑t ∧ s ⊆ ↑t' ∧ t'.card = (s' ∪ t).card :=
    fun t =>
    Finset.induction_on t
      (fun s' hs' _ hss' =>
        have : s = ↑s' := eq_of_linearIndependent_of_span_subtype hs hs' <| by simpa using hss'
        ⟨s', by simp [this]⟩)
      fun b₁ t hb₁t ih s' hs' hst hss' =>
      have hb₁s : b₁ ∉ s := fun h => by
        have : b₁ ∈ s ∩ ↑(insert b₁ t) := ⟨h, Finset.mem_insert_self _ _⟩
        rwa [hst] at this
      have hb₁s' : b₁ ∉ s' := fun h => hb₁s <| hs' h
      have hst : s ∩ ↑t = ∅ :=
        eq_empty_of_subset_empty <|
          -- Porting note: `-inter_subset_left, -subset_inter_iff` required.
          Subset.trans
            (by simp [inter_subset_inter, Subset.refl, -inter_subset_left, -subset_inter_iff])
            (le_of_eq hst)
      Classical.by_cases (p := s ⊆ (span K ↑(s' ∪ t) : Submodule K V))
        (fun this =>
          let ⟨u, hust, hsu, Eq⟩ := ih _ hs' hst this
          have hb₁u : b₁ ∉ u := fun h => (hust h).elim hb₁s hb₁t
          ⟨insert b₁ u, by simp [insert_subset_insert hust], Subset.trans hsu (by simp), by
            simp [Eq, hb₁t, hb₁s', hb₁u]⟩)
        fun this =>
        let ⟨b₂, hb₂s, hb₂t⟩ := not_subset.mp this
        have hb₂t' : b₂ ∉ s' ∪ t := fun h => hb₂t <| subset_span h
        have : s ⊆ (span K ↑(insert b₂ s' ∪ t) : Submodule K V) := fun b₃ hb₃ => by
          have : ↑(s' ∪ insert b₁ t) ⊆ insert b₁ (insert b₂ ↑(s' ∪ t) : Set V) := by
            -- Porting note: Too many theorems to be excluded, so
            --               `simp only` is shorter.
            simp only [insert_eq, union_subset_union, Subset.refl,
              subset_union_right, Finset.union_insert, Finset.coe_insert]
          have hb₃ : b₃ ∈ span K (insert b₁ (insert b₂ ↑(s' ∪ t) : Set V)) :=
            span_mono this (hss' hb₃)
          have : s ⊆ (span K (insert b₁ ↑(s' ∪ t)) : Submodule K V) := by
            simpa [insert_eq, -singleton_union, -union_singleton] using hss'
          -- Porting note: `by exact` is required to prevent timeout.
          have hb₁ : b₁ ∈ span K (insert b₂ ↑(s' ∪ t)) := by
            exact mem_span_insert_exchange (this hb₂s) hb₂t
          rw [span_insert_eq_span hb₁] at hb₃; simpa using hb₃
        let ⟨u, hust, hsu, eq⟩ := ih _ (by simp [insert_subset_iff, hb₂s, hs']) hst this
        -- Porting note: `hb₂t'` → `Finset.card_insert_of_not_mem hb₂t'`
        ⟨u, Subset.trans hust <| union_subset_union (Subset.refl _) (by simp [subset_insert]), hsu,
          by simp [eq, Finset.card_insert_of_not_mem hb₂t', hb₁t, hb₁s']⟩
  have eq : ((t.filter fun x => x ∈ s) ∪ t.filter fun x => x ∉ s) = t := by
    ext1 x
    by_cases x ∈ s <;> simp [*]
  apply
    Exists.elim
      (this (t.filter fun x => x ∉ s) (t.filter fun x => x ∈ s) (by simp [Set.subset_def])
        (by simp +contextual [Set.ext_iff]) (by rwa [eq]))
  intro u h
  exact
    ⟨u, Subset.trans h.1 (by simp +contextual [subset_def, and_imp, or_imp]),
      h.2.1, by simp only [h.2.2, eq]⟩


theorem exists_finite_card_le_of_finite_of_linearIndependent_of_span (ht : t.Finite)
    (hs : LinearIndependent K (fun x => x : s → V)) (hst : s ⊆ span K t) :
    ∃ h : s.Finite, h.toFinset.card ≤ ht.toFinset.card :=
                                                         /-
                                                           K : Type u_3
                                                           V : Type u
                                                           inst✝² : DivisionRing K
                                                           inst✝¹ : AddCommGroup V
                                                           inst✝ : Module K V
                                                           s t : Set V
                                                           ht : t.Finite
                                                           hs : LinearIndependent K fun x => ↑x
                                                           hst : HasSubset.Subset s ↑(Submodule.span K t)
                                                           ⊢ HasSubset.Subset s ↑(Submodule.span K ↑ht.toFinset)
                                                         -/
  have : s ⊆ (span K ↑ht.toFinset : Submodule K V) := by simpa
                                                         /-
                                                           🎉 no goals
                                                         -/
  let ⟨u, _hust, hsu, Eq⟩ := exists_of_linearIndependent_of_finite_span hs this
  have : s.Finite := u.finite_toSet.subset hsu
            /-
              K : Type u_3
              V : Type u
              inst✝² : DivisionRing K
              inst✝¹ : AddCommGroup V
              inst✝ : Module K V
              s t : Set V
              ht : t.Finite
              hs : LinearIndependent K fun x => ↑x
              hst : HasSubset.Subset s ↑(Submodule.span K t)
              this✝ : HasSubset.Subset s ↑(Submodule.span K ↑ht.toFinset)
              u : Finset V
              _hust : HasSubset.Subset (↑u) (Union.union s ↑ht.toFinset)
              hsu : HasSubset.Subset s ↑u
              Eq : _root_.Eq u.card ht.toFinset.card
              this : s.Finite
              ⊢ LE.le this.toFinset.card ht.toFinset.card
            -/
  ⟨this, by rw [← Eq]; exact Finset.card_le_card <| Finset.coe_subset.mp <| by simp [hsu]⟩
                       /-
                         🎉 no goals
                       -/


