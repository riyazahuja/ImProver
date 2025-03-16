/-- Turn a function `f : C(X, ℝ)` into a continuous map into `Set.Icc (-‖f‖) (‖f‖)`,
thereby explicitly attaching bounds.
-/
def attachBound (f : C(X, ℝ)) : C(X, Set.Icc (-‖f‖) ‖f‖) where
  toFun x := ⟨f x, ⟨neg_norm_le_apply f x, apply_le_norm f x⟩⟩


@[simp]
theorem attachBound_apply_coe (f : C(X, ℝ)) (x : X) : ((attachBound f) x : ℝ) = f x :=
  rfl


theorem polynomial_comp_attachBound (A : Subalgebra ℝ C(X, ℝ)) (f : A) (g : ℝ[X]) :
    (g.toContinuousMapOn (Set.Icc (-‖f‖) ‖f‖)).comp (f : C(X, ℝ)).attachBound =
      Polynomial.aeval f g := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f : Subtype fun x => Membership.mem A x
    g : Polynomial Real
    ⊢ Eq ((g.toContinuousMapOn (Set.Icc (Neg.neg (Norm.norm f)) (Norm.norm f))).co …
  -/
  ext
  simp only [ContinuousMap.coe_comp, Function.comp_apply, ContinuousMap.attachBound_apply_coe,
    Polynomial.toContinuousMapOn_apply, Polynomial.aeval_subalgebra_coe,
    Polynomial.aeval_continuousMap_apply, Polynomial.toContinuousMap_apply]
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f : Subtype fun x => Membership.mem A x
    g : Polynomial Real
    a✝ : X
    ⊢ Eq (Polynomial.eval (↑((↑f).attachBound a✝)) g) (Polynomial.eval (↑f a✝) g)
  -/
  erw [ContinuousMap.attachBound_apply_coe]
  /-
    🎉 no goals
  -/


/-- Given a continuous function `f` in a subalgebra of `C(X, ℝ)`, postcomposing by a polynomial
gives another function in `A`.

This lemma proves something slightly more subtle than this:
we take `f`, and think of it as a function into the restricted target `Set.Icc (-‖f‖) ‖f‖)`,
and then postcompose with a polynomial function on that interval.
This is in fact the same situation as above, and so also gives a function in `A`.
-/
theorem polynomial_comp_attachBound_mem (A : Subalgebra ℝ C(X, ℝ)) (f : A) (g : ℝ[X]) :
    (g.toContinuousMapOn (Set.Icc (-‖f‖) ‖f‖)).comp (f : C(X, ℝ)).attachBound ∈ A := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f : Subtype fun x => Membership.mem A x
    g : Polynomial Real
    ⊢ Membership.mem A ((g.toContinuousMapOn (Set.Icc (Neg.neg (Norm.norm f)) (Nor …
  -/
  rw [polynomial_comp_attachBound]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f : Subtype fun x => Membership.mem A x
    g : Polynomial Real
    ⊢ Membership.mem A ↑((Polynomial.aeval f) g)
  -/
  apply SetLike.coe_mem
  /-
    🎉 no goals
  -/


theorem comp_attachBound_mem_closure (A : Subalgebra ℝ C(X, ℝ)) (f : A)
    (p : C(Set.Icc (-‖f‖) ‖f‖, ℝ)) : p.comp (attachBound (f : C(X, ℝ))) ∈ A.topologicalClosure := by
  -- `p` itself is in the closure of polynomials, by the Weierstrass theorem,
  have mem_closure : p ∈ (polynomialFunctions (Set.Icc (-‖f‖) ‖f‖)).topologicalClosure :=
    continuousMap_mem_polynomialFunctions_closure _ _ p
  -- and so there are polynomials arbitrarily close.
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f : Subtype fun x => Membership.mem A x
    p : ContinuousMap (↑(Set.Icc (Neg.neg (Norm.norm f)) (Norm.norm f))) Real
    mem_closure : Membership.mem (polynomialFunctions (Set.Icc (Neg.neg (Norm.norm …
    ⊢ Membership.mem A.topologicalClosure (p.comp (↑f).attachBound)
  -/
  have frequently_mem_polynomials := mem_closure_iff_frequently.mp mem_closure
  -- To prove `p.comp (attachBound f)` is in the closure of `A`,
  -- we show there are elements of `A` arbitrarily close.
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f : Subtype fun x => Membership.mem A x
    p : ContinuousMap (↑(Set.Icc (Neg.neg (Norm.norm f)) (Norm.norm f))) Real
    mem_closure : Membership.mem (polynomialFunctions (Set.Icc (Neg.neg (Norm.norm …
    frequently_mem_polynomials : Filter.Frequently (fun x => Membership.mem (↑(pol …
    ⊢ Membership.mem A.topologicalClosure (p.comp (↑f).attachBound)
  -/
  apply mem_closure_iff_frequently.mpr
  -- To show that, we pull back the polynomials close to `p`,
  refine
    ((compRightContinuousMap ℝ (attachBound (f : C(X, ℝ)))).continuousAt
            p).tendsto.frequently_map
      _ ?_ frequently_mem_polynomials
  -- but need to show that those pullbacks are actually in `A`.
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f : Subtype fun x => Membership.mem A x
    p : ContinuousMap (↑(Set.Icc (Neg.neg (Norm.norm f)) (Norm.norm f))) Real
    mem_closure : Membership.mem (polynomialFunctions (Set.Icc (Neg.neg (Norm.norm …
    frequently_mem_polynomials : Filter.Frequently (fun x => Membership.mem (↑(pol …
    ⊢ ∀ (x : ContinuousMap (↑(Set.Icc (Neg.neg (Norm.norm ↑f)) (Norm.norm ↑f))) Re …
  -/
  rintro _ ⟨g, ⟨-, rfl⟩⟩
  simp only [SetLike.mem_coe, AlgHom.coe_toRingHom, compRightContinuousMap_apply,
    Polynomial.toContinuousMapOnAlgHom_apply]
  /-
    case intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f : Subtype fun x => Membership.mem A x
    p : ContinuousMap (↑(Set.Icc (Neg.neg (Norm.norm f)) (Norm.norm f))) Real
    mem_closure : Membership.mem (polynomialFunctions (Set.Icc (Neg.neg (Norm.norm …
    frequently_mem_polynomials : Filter.Frequently (fun x => Membership.mem (↑(pol …
    g : Polynomial Real
    ⊢ Membership.mem A.toSubsemiring ((g.toContinuousMapOn (Set.Icc (Neg.neg (Norm …
  -/
  apply polynomial_comp_attachBound_mem
  /-
    🎉 no goals
  -/


theorem abs_mem_subalgebra_closure (A : Subalgebra ℝ C(X, ℝ)) (f : A) :
    |(f : C(X, ℝ))| ∈ A.topologicalClosure := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f : Subtype fun x => Membership.mem A x
    ⊢ Membership.mem A.topologicalClosure (abs ↑f)
  -/
  let f' := attachBound (f : C(X, ℝ))
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f : Subtype fun x => Membership.mem A x
    f' : ContinuousMap X ↑(Set.Icc (Neg.neg (Norm.norm ↑f)) (Norm.norm ↑f)) := (↑f …
    ⊢ Membership.mem A.topologicalClosure (abs ↑f)
  -/
  let abs : C(Set.Icc (-‖f‖) ‖f‖, ℝ) := { toFun := fun x : Set.Icc (-‖f‖) ‖f‖ => |(x : ℝ)| }
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f : Subtype fun x => Membership.mem A x
    f' : ContinuousMap X ↑(Set.Icc (Neg.neg (Norm.norm ↑f)) (Norm.norm ↑f)) := (↑f …
    abs : ContinuousMap (↑(Set.Icc (Neg.neg (Norm.norm f)) (Norm.norm f))) Real := …
    ⊢ Membership.mem A.topologicalClosure (_root_.abs ↑f)
  -/
  change abs.comp f' ∈ A.topologicalClosure
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f : Subtype fun x => Membership.mem A x
    f' : ContinuousMap X ↑(Set.Icc (Neg.neg (Norm.norm ↑f)) (Norm.norm ↑f)) := (↑f …
    abs : ContinuousMap (↑(Set.Icc (Neg.neg (Norm.norm f)) (Norm.norm f))) Real := …
    ⊢ Membership.mem A.topologicalClosure (abs.comp f')
  -/
  apply comp_attachBound_mem_closure
  /-
    🎉 no goals
  -/


theorem inf_mem_subalgebra_closure (A : Subalgebra ℝ C(X, ℝ)) (f g : A) :
    (f : C(X, ℝ)) ⊓ (g : C(X, ℝ)) ∈ A.topologicalClosure := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f g : Subtype fun x => Membership.mem A x
    ⊢ Membership.mem A.topologicalClosure (Min.min ↑f ↑g)
  -/
  rw [inf_eq_half_smul_add_sub_abs_sub' ℝ]
  refine
    A.topologicalClosure.smul_mem
      (A.topologicalClosure.sub_mem
        (A.topologicalClosure.add_mem (A.le_topologicalClosure f.property)
          (A.le_topologicalClosure g.property))
        ?_)
      _
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f g : Subtype fun x => Membership.mem A x
    ⊢ Membership.mem A.topologicalClosure (abs (HSub.hSub ↑g ↑f))
  -/
  exact mod_cast abs_mem_subalgebra_closure A _
  /-
    🎉 no goals
  -/


theorem inf_mem_closed_subalgebra (A : Subalgebra ℝ C(X, ℝ)) (h : IsClosed (A : Set C(X, ℝ)))
    (f g : A) : (f : C(X, ℝ)) ⊓ (g : C(X, ℝ)) ∈ A := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    h : IsClosed ↑A
    f g : Subtype fun x => Membership.mem A x
    ⊢ Membership.mem A (Min.min ↑f ↑g)
  -/
  convert inf_mem_subalgebra_closure A f g
  /-
    case h.e'_4
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    h : IsClosed ↑A
    f g : Subtype fun x => Membership.mem A x
    ⊢ Eq A A.topologicalClosure
  -/
  apply SetLike.ext'
  /-
    case h.e'_4.h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    h : IsClosed ↑A
    f g : Subtype fun x => Membership.mem A x
    ⊢ Eq ↑A ↑A.topologicalClosure
  -/
  symm
  /-
    case h.e'_4.h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    h : IsClosed ↑A
    f g : Subtype fun x => Membership.mem A x
    ⊢ Eq ↑A.topologicalClosure ↑A
  -/
  erw [closure_eq_iff_isClosed]
  /-
    case h.e'_4.h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    h : IsClosed ↑A
    f g : Subtype fun x => Membership.mem A x
    ⊢ IsClosed ↑A.toSubsemiring
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem sup_mem_subalgebra_closure (A : Subalgebra ℝ C(X, ℝ)) (f g : A) :
    (f : C(X, ℝ)) ⊔ (g : C(X, ℝ)) ∈ A.topologicalClosure := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f g : Subtype fun x => Membership.mem A x
    ⊢ Membership.mem A.topologicalClosure (Max.max ↑f ↑g)
  -/
  rw [sup_eq_half_smul_add_add_abs_sub' ℝ]
  refine
    A.topologicalClosure.smul_mem
      (A.topologicalClosure.add_mem
        (A.topologicalClosure.add_mem (A.le_topologicalClosure f.property)
          (A.le_topologicalClosure g.property))
        ?_)
      _
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    f g : Subtype fun x => Membership.mem A x
    ⊢ Membership.mem A.topologicalClosure (abs (HSub.hSub ↑g ↑f))
  -/
  exact mod_cast abs_mem_subalgebra_closure A _
  /-
    🎉 no goals
  -/


theorem sup_mem_closed_subalgebra (A : Subalgebra ℝ C(X, ℝ)) (h : IsClosed (A : Set C(X, ℝ)))
    (f g : A) : (f : C(X, ℝ)) ⊔ (g : C(X, ℝ)) ∈ A := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    h : IsClosed ↑A
    f g : Subtype fun x => Membership.mem A x
    ⊢ Membership.mem A (Max.max ↑f ↑g)
  -/
  convert sup_mem_subalgebra_closure A f g
  /-
    case h.e'_4
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    h : IsClosed ↑A
    f g : Subtype fun x => Membership.mem A x
    ⊢ Eq A A.topologicalClosure
  -/
  apply SetLike.ext'
  /-
    case h.e'_4.h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    h : IsClosed ↑A
    f g : Subtype fun x => Membership.mem A x
    ⊢ Eq ↑A ↑A.topologicalClosure
  -/
  symm
  /-
    case h.e'_4.h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    h : IsClosed ↑A
    f g : Subtype fun x => Membership.mem A x
    ⊢ Eq ↑A.topologicalClosure ↑A
  -/
  erw [closure_eq_iff_isClosed]
  /-
    case h.e'_4.h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    h : IsClosed ↑A
    f g : Subtype fun x => Membership.mem A x
    ⊢ IsClosed ↑A.toSubsemiring
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem sublattice_closure_eq_top (L : Set C(X, ℝ)) (nA : L.Nonempty)
    (inf_mem : ∀ᵉ (f ∈ L) (g ∈ L), f ⊓ g ∈ L)
    (sup_mem : ∀ᵉ (f ∈ L) (g ∈ L), f ⊔ g ∈ L) (sep : L.SeparatesPointsStrongly) :
    closure L = ⊤ := by
  -- We start by boiling down to a statement about close approximation.
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : L.SeparatesPointsStrongly
    ⊢ Eq (closure L) Top.top
  -/
  rw [eq_top_iff]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : L.SeparatesPointsStrongly
    ⊢ LE.le Top.top (closure L)
  -/
  rintro f -
  refine
    Filter.Frequently.mem_closure
      ((Filter.HasBasis.frequently_iff Metric.nhds_basis_ball).mpr fun ε pos => ?_)
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : L.SeparatesPointsStrongly
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    ⊢ Exists fun x => And (Membership.mem (Metric.ball f ε) x) (Membership.mem L x)
  -/
  simp only [exists_prop, Metric.mem_ball]
  -- It will be helpful to assume `X` is nonempty later,
  -- so we get that out of the way here.
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : L.SeparatesPointsStrongly
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
  -/
  by_cases nX : Nonempty X
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : L.SeparatesPointsStrongly
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
  -/
  swap
    /-
      case neg
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : CompactSpace X
      L : Set (ContinuousMap X Real)
      nA : L.Nonempty
      inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sep : L.SeparatesPointsStrongly
      f : ContinuousMap X Real
      ε : Real
      pos : LT.lt 0 ε
      nX : Not (Nonempty X)
      ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
    -/
  · exact ⟨nA.some, (dist_lt_iff pos).mpr fun x => False.elim (nX ⟨x⟩), nA.choose_spec⟩
    /-
      🎉 no goals
    -/
  /-
    The strategy now is to pick a family of continuous functions `g x y` in `A`
    with the property that `g x y x = f x` and `g x y y = f y`
    (this is immediate from `h : SeparatesPointsStrongly`)
    then use continuity to see that `g x y` is close to `f` near both `x` and `y`,
    and finally using compactness to produce the desired function `h`
    as a maximum over finitely many `x` of a minimum over finitely many `y` of the `g x y`.
    -/
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : L.SeparatesPointsStrongly
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
  -/
  dsimp only [Set.SeparatesPointsStrongly] at sep
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
  -/
  choose g hg w₁ w₂ using sep f
  -- For each `x y`, we define `U x y` to be `{z | f z - ε < g x y z}`,
  -- and observe this is a neighbourhood of `y`.
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    g : X → X → ContinuousMap X Real
    hg : ∀ (x y : X), Membership.mem L (g x y)
    w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
    w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
    ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
  -/
  let U : X → X → Set X := fun x y => {z | f z - ε < g x y z}
  have U_nhd_y : ∀ x y, U x y ∈ 𝓝 y := by
    intro x y
    refine IsOpen.mem_nhds ?_ ?_
    · apply isOpen_lt <;> continuity
    · rw [Set.mem_setOf_eq, w₂]
      exact sub_lt_self _ pos
  -- Fixing `x` for a moment, we have a family of functions `fun y ↦ g x y`
  -- which on different patches (the `U x y`) are greater than `f z - ε`.
  -- Taking the supremum of these functions
  -- indexed by a finite collection of patches which cover `X`
  -- will give us an element of `A` that is globally greater than `f z - ε`
  -- and still equal to `f x` at `x`.
  -- Since `X` is compact, for every `x` there is some finset `ys t`
  -- so the union of the `U x y` for `y ∈ ys x` still covers everything.
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    g : X → X → ContinuousMap X Real
    hg : ∀ (x y : X), Membership.mem L (g x y)
    w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
    w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
    U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
    U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
    ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
  -/
  let ys : X → Finset X := fun x => (CompactSpace.elim_nhds_subcover (U x) (U_nhd_y x)).choose
  let ys_w : ∀ x, ⋃ y ∈ ys x, U x y = ⊤ := fun x =>
    (CompactSpace.elim_nhds_subcover (U x) (U_nhd_y x)).choose_spec
  have ys_nonempty : ∀ x, (ys x).Nonempty := fun x =>
    Set.nonempty_of_union_eq_top_of_nonempty _ _ nX (ys_w x)
  -- Thus for each `x` we have the desired `h x : A` so `f z - ε < h x z` everywhere
  -- and `h x x = f x`.
  let h : X → L := fun x =>
    ⟨(ys x).sup' (ys_nonempty x) fun y => (g x y : C(X, ℝ)),
      Finset.sup'_mem _ sup_mem _ _ _ fun y _ => hg x y⟩
  have lt_h : ∀ x z, f z - ε < (h x : X → ℝ) z := by
    intro x z
    obtain ⟨y, ym, zm⟩ := Set.exists_set_mem_of_union_eq_top _ _ (ys_w x) z
    dsimp [h]
    simp only [Subtype.coe_mk, coe_sup', Finset.sup'_apply, Finset.lt_sup'_iff]
    exact ⟨y, ym, zm⟩
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    g : X → X → ContinuousMap X Real
    hg : ∀ (x y : X), Membership.mem L (g x y)
    w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
    w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
    U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
    U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
    ys : X → Finset X := fun x => ⋯.choose
    ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
    ys_nonempty : ∀ (x : X), (ys x).Nonempty
    h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
    lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
    ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
  -/
  have h_eq : ∀ x, (h x : X → ℝ) x = f x := by intro x; simp [h, w₁]
  -- For each `x`, we define `W x` to be `{z | h x z < f z + ε}`,
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    g : X → X → ContinuousMap X Real
    hg : ∀ (x y : X), Membership.mem L (g x y)
    w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
    w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
    U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
    U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
    ys : X → Finset X := fun x => ⋯.choose
    ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
    ys_nonempty : ∀ (x : X), (ys x).Nonempty
    h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
    lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
    h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
    ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
  -/
  let W : X → Set X := fun x => {z | (h x : X → ℝ) z < f z + ε}
  -- This is still a neighbourhood of `x`.
  have W_nhd : ∀ x, W x ∈ 𝓝 x := by
    intro x
    refine IsOpen.mem_nhds ?_ ?_
    · apply isOpen_lt <;> fun_prop
    · dsimp only [W, Set.mem_setOf_eq]
      rw [h_eq]
      exact lt_add_of_pos_right _ pos
  -- Since `X` is compact, there is some finset `ys t`
  -- so the union of the `W x` for `x ∈ xs` still covers everything.
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    g : X → X → ContinuousMap X Real
    hg : ∀ (x y : X), Membership.mem L (g x y)
    w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
    w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
    U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
    U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
    ys : X → Finset X := fun x => ⋯.choose
    ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
    ys_nonempty : ∀ (x : X), (ys x).Nonempty
    h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
    lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
    h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
    W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
    W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
    ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
  -/
  let xs : Finset X := (CompactSpace.elim_nhds_subcover W W_nhd).choose
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    g : X → X → ContinuousMap X Real
    hg : ∀ (x y : X), Membership.mem L (g x y)
    w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
    w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
    U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
    U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
    ys : X → Finset X := fun x => ⋯.choose
    ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
    ys_nonempty : ∀ (x : X), (ys x).Nonempty
    h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
    lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
    h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
    W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
    W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
    xs : Finset X := ⋯.choose
    ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
  -/
  let xs_w : ⋃ x ∈ xs, W x = ⊤ := (CompactSpace.elim_nhds_subcover W W_nhd).choose_spec
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    g : X → X → ContinuousMap X Real
    hg : ∀ (x y : X), Membership.mem L (g x y)
    w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
    w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
    U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
    U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
    ys : X → Finset X := fun x => ⋯.choose
    ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
    ys_nonempty : ∀ (x : X), (ys x).Nonempty
    h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
    lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
    h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
    W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
    W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
    xs : Finset X := ⋯.choose
    xs_w : Eq (Set.iUnion fun x => Set.iUnion fun h => W x) Top.top := Exists.choo …
    ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
  -/
  have xs_nonempty : xs.Nonempty := Set.nonempty_of_union_eq_top_of_nonempty _ _ nX xs_w
  -- Finally our candidate function is the infimum over `x ∈ xs` of the `h x`.
  -- This function is then globally less than `f z + ε`.
  let k : (L : Type _) :=
    ⟨xs.inf' xs_nonempty fun x => (h x : C(X, ℝ)),
      Finset.inf'_mem _ inf_mem _ _ _ fun x _ => (h x).2⟩
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    g : X → X → ContinuousMap X Real
    hg : ∀ (x y : X), Membership.mem L (g x y)
    w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
    w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
    U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
    U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
    ys : X → Finset X := fun x => ⋯.choose
    ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
    ys_nonempty : ∀ (x : X), (ys x).Nonempty
    h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
    lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
    h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
    W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
    W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
    xs : Finset X := ⋯.choose
    xs_w : Eq (Set.iUnion fun x => Set.iUnion fun h => W x) Top.top := Exists.choo …
    xs_nonempty : xs.Nonempty
    k : ↑L := ⟨xs.inf' xs_nonempty fun x => ↑(h x), ⋯⟩
    ⊢ Exists fun x => And (LT.lt (Dist.dist x f) ε) (Membership.mem L x)
  -/
  refine ⟨k.1, ?_, k.2⟩
  -- We just need to verify the bound, which we do pointwise.
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    g : X → X → ContinuousMap X Real
    hg : ∀ (x y : X), Membership.mem L (g x y)
    w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
    w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
    U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
    U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
    ys : X → Finset X := fun x => ⋯.choose
    ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
    ys_nonempty : ∀ (x : X), (ys x).Nonempty
    h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
    lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
    h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
    W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
    W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
    xs : Finset X := ⋯.choose
    xs_w : Eq (Set.iUnion fun x => Set.iUnion fun h => W x) Top.top := Exists.choo …
    xs_nonempty : xs.Nonempty
    k : ↑L := ⟨xs.inf' xs_nonempty fun x => ↑(h x), ⋯⟩
    ⊢ LT.lt (Dist.dist (↑k) f) ε
  -/
  rw [dist_lt_iff pos]
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    g : X → X → ContinuousMap X Real
    hg : ∀ (x y : X), Membership.mem L (g x y)
    w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
    w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
    U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
    U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
    ys : X → Finset X := fun x => ⋯.choose
    ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
    ys_nonempty : ∀ (x : X), (ys x).Nonempty
    h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
    lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
    h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
    W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
    W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
    xs : Finset X := ⋯.choose
    xs_w : Eq (Set.iUnion fun x => Set.iUnion fun h => W x) Top.top := Exists.choo …
    xs_nonempty : xs.Nonempty
    k : ↑L := ⟨xs.inf' xs_nonempty fun x => ↑(h x), ⋯⟩
    ⊢ ∀ (x : X), LT.lt (Dist.dist (↑k x) (f x)) ε
  -/
  intro z
  -- We rewrite into this particular form,
  -- so that simp lemmas about inequalities involving `Finset.inf'` can fire.
  rw [show ∀ a b ε : ℝ, dist a b < ε ↔ a < b + ε ∧ b - ε < a by
        intros; simp only [← Metric.mem_ball, Real.ball_eq_Ioo, Set.mem_Ioo, and_comm]]
  /-
    case pos
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    L : Set (ContinuousMap X Real)
    nA : L.Nonempty
    inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
    sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    nX : Nonempty X
    g : X → X → ContinuousMap X Real
    hg : ∀ (x y : X), Membership.mem L (g x y)
    w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
    w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
    U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
    U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
    ys : X → Finset X := fun x => ⋯.choose
    ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
    ys_nonempty : ∀ (x : X), (ys x).Nonempty
    h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
    lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
    h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
    W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
    W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
    xs : Finset X := ⋯.choose
    xs_w : Eq (Set.iUnion fun x => Set.iUnion fun h => W x) Top.top := Exists.choo …
    xs_nonempty : xs.Nonempty
    k : ↑L := ⟨xs.inf' xs_nonempty fun x => ↑(h x), ⋯⟩
    z : X
    ⊢ And (LT.lt (↑k z) (HAdd.hAdd (f z) ε)) (LT.lt (HSub.hSub (f z) ε) (↑k z))
  -/
  fconstructor
    /-
      case pos.left
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : CompactSpace X
      L : Set (ContinuousMap X Real)
      nA : L.Nonempty
      inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
      f : ContinuousMap X Real
      ε : Real
      pos : LT.lt 0 ε
      nX : Nonempty X
      g : X → X → ContinuousMap X Real
      hg : ∀ (x y : X), Membership.mem L (g x y)
      w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
      w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
      U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
      U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
      ys : X → Finset X := fun x => ⋯.choose
      ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
      ys_nonempty : ∀ (x : X), (ys x).Nonempty
      h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
      lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
      h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
      W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
      W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
      xs : Finset X := ⋯.choose
      xs_w : Eq (Set.iUnion fun x => Set.iUnion fun h => W x) Top.top := Exists.choo …
      xs_nonempty : xs.Nonempty
      k : ↑L := ⟨xs.inf' xs_nonempty fun x => ↑(h x), ⋯⟩
      z : X
      ⊢ LT.lt (↑k z) (HAdd.hAdd (f z) ε)
    -/
  · dsimp
    /-
      case pos.left
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : CompactSpace X
      L : Set (ContinuousMap X Real)
      nA : L.Nonempty
      inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
      f : ContinuousMap X Real
      ε : Real
      pos : LT.lt 0 ε
      nX : Nonempty X
      g : X → X → ContinuousMap X Real
      hg : ∀ (x y : X), Membership.mem L (g x y)
      w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
      w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
      U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
      U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
      ys : X → Finset X := fun x => ⋯.choose
      ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
      ys_nonempty : ∀ (x : X), (ys x).Nonempty
      h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
      lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
      h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
      W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
      W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
      xs : Finset X := ⋯.choose
      xs_w : Eq (Set.iUnion fun x => Set.iUnion fun h => W x) Top.top := Exists.choo …
      xs_nonempty : xs.Nonempty
      k : ↑L := ⟨xs.inf' xs_nonempty fun x => ↑(h x), ⋯⟩
      z : X
      ⊢ LT.lt ((xs.inf' xs_nonempty fun x => (ys x).sup' ⋯ fun y => g x y) z) (HAdd. …
    -/
    simp only [k, Finset.inf'_lt_iff, ContinuousMap.inf'_apply]
    /-
      case pos.left
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : CompactSpace X
      L : Set (ContinuousMap X Real)
      nA : L.Nonempty
      inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
      f : ContinuousMap X Real
      ε : Real
      pos : LT.lt 0 ε
      nX : Nonempty X
      g : X → X → ContinuousMap X Real
      hg : ∀ (x y : X), Membership.mem L (g x y)
      w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
      w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
      U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
      U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
      ys : X → Finset X := fun x => ⋯.choose
      ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
      ys_nonempty : ∀ (x : X), (ys x).Nonempty
      h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
      lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
      h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
      W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
      W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
      xs : Finset X := ⋯.choose
      xs_w : Eq (Set.iUnion fun x => Set.iUnion fun h => W x) Top.top := Exists.choo …
      xs_nonempty : xs.Nonempty
      k : ↑L := ⟨xs.inf' xs_nonempty fun x => ↑(h x), ⋯⟩
      z : X
      ⊢ Exists fun i => And (Membership.mem xs i) (LT.lt (((ys i).sup' ⋯ fun y => g  …
    -/
    exact Set.exists_set_mem_of_union_eq_top _ _ xs_w z
    /-
      🎉 no goals
    -/
    /-
      case pos.right
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : CompactSpace X
      L : Set (ContinuousMap X Real)
      nA : L.Nonempty
      inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
      f : ContinuousMap X Real
      ε : Real
      pos : LT.lt 0 ε
      nX : Nonempty X
      g : X → X → ContinuousMap X Real
      hg : ∀ (x y : X), Membership.mem L (g x y)
      w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
      w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
      U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
      U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
      ys : X → Finset X := fun x => ⋯.choose
      ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
      ys_nonempty : ∀ (x : X), (ys x).Nonempty
      h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
      lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
      h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
      W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
      W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
      xs : Finset X := ⋯.choose
      xs_w : Eq (Set.iUnion fun x => Set.iUnion fun h => W x) Top.top := Exists.choo …
      xs_nonempty : xs.Nonempty
      k : ↑L := ⟨xs.inf' xs_nonempty fun x => ↑(h x), ⋯⟩
      z : X
      ⊢ LT.lt (HSub.hSub (f z) ε) (↑k z)
    -/
  · dsimp
    /-
      case pos.right
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : CompactSpace X
      L : Set (ContinuousMap X Real)
      nA : L.Nonempty
      inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
      f : ContinuousMap X Real
      ε : Real
      pos : LT.lt 0 ε
      nX : Nonempty X
      g : X → X → ContinuousMap X Real
      hg : ∀ (x y : X), Membership.mem L (g x y)
      w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
      w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
      U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
      U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
      ys : X → Finset X := fun x => ⋯.choose
      ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
      ys_nonempty : ∀ (x : X), (ys x).Nonempty
      h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
      lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
      h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
      W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
      W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
      xs : Finset X := ⋯.choose
      xs_w : Eq (Set.iUnion fun x => Set.iUnion fun h => W x) Top.top := Exists.choo …
      xs_nonempty : xs.Nonempty
      k : ↑L := ⟨xs.inf' xs_nonempty fun x => ↑(h x), ⋯⟩
      z : X
      ⊢ LT.lt (HSub.hSub (f z) ε) ((xs.inf' xs_nonempty fun x => (ys x).sup' ⋯ fun y …
    -/
    simp only [k, Finset.lt_inf'_iff, ContinuousMap.inf'_apply]
    /-
      case pos.right
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : CompactSpace X
      L : Set (ContinuousMap X Real)
      nA : L.Nonempty
      inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
      f : ContinuousMap X Real
      ε : Real
      pos : LT.lt 0 ε
      nX : Nonempty X
      g : X → X → ContinuousMap X Real
      hg : ∀ (x y : X), Membership.mem L (g x y)
      w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
      w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
      U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
      U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
      ys : X → Finset X := fun x => ⋯.choose
      ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
      ys_nonempty : ∀ (x : X), (ys x).Nonempty
      h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
      lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
      h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
      W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
      W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
      xs : Finset X := ⋯.choose
      xs_w : Eq (Set.iUnion fun x => Set.iUnion fun h => W x) Top.top := Exists.choo …
      xs_nonempty : xs.Nonempty
      k : ↑L := ⟨xs.inf' xs_nonempty fun x => ↑(h x), ⋯⟩
      z : X
      ⊢ ∀ (i : X), Membership.mem xs i → LT.lt (HSub.hSub (f z) ε) (((ys i).sup' ⋯ f …
    -/
    rintro x -
    /-
      case pos.right
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : CompactSpace X
      L : Set (ContinuousMap X Real)
      nA : L.Nonempty
      inf_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sup_mem : ∀ (f : ContinuousMap X Real), Membership.mem L f → ∀ (g : Continuous …
      sep : ∀ (v : X → Real) (x y : X), Exists fun f => And (Membership.mem L f) (An …
      f : ContinuousMap X Real
      ε : Real
      pos : LT.lt 0 ε
      nX : Nonempty X
      g : X → X → ContinuousMap X Real
      hg : ∀ (x y : X), Membership.mem L (g x y)
      w₁ : ∀ (x y : X), Eq ((g x y) x) (f x)
      w₂ : ∀ (x y : X), Eq ((g x y) y) (f y)
      U : X → X → Set X := fun x y => setOf fun z => LT.lt (HSub.hSub (f z) ε) ((g x …
      U_nhd_y : ∀ (x y : X), Membership.mem (nhds y) (U x y)
      ys : X → Finset X := fun x => ⋯.choose
      ys_w : ∀ (x : X), Eq (Set.iUnion fun y => Set.iUnion fun h => U x y) Top.top : …
      ys_nonempty : ∀ (x : X), (ys x).Nonempty
      h : X → ↑L := fun x => ⟨(ys x).sup' ⋯ fun y => g x y, ⋯⟩
      lt_h : ∀ (x z : X), LT.lt (HSub.hSub (f z) ε) (↑(h x) z)
      h_eq : ∀ (x : X), Eq (↑(h x) x) (f x)
      W : X → Set X := fun x => setOf fun z => LT.lt (↑(h x) z) (HAdd.hAdd (f z) ε)
      W_nhd : ∀ (x : X), Membership.mem (nhds x) (W x)
      xs : Finset X := ⋯.choose
      xs_w : Eq (Set.iUnion fun x => Set.iUnion fun h => W x) Top.top := Exists.choo …
      xs_nonempty : xs.Nonempty
      k : ↑L := ⟨xs.inf' xs_nonempty fun x => ↑(h x), ⋯⟩
      z x : X
      ⊢ LT.lt (HSub.hSub (f z) ε) (((ys x).sup' ⋯ fun y => g x y) z)
    -/
    apply lt_h
    /-
      🎉 no goals
    -/


/-- The **Stone-Weierstrass Approximation Theorem**,
that a subalgebra `A` of `C(X, ℝ)`, where `X` is a compact topological space,
is dense if it separates points.
-/
theorem subalgebra_topologicalClosure_eq_top_of_separatesPoints (A : Subalgebra ℝ C(X, ℝ))
    (w : A.SeparatesPoints) : A.topologicalClosure = ⊤ := by
  -- The closure of `A` is closed under taking `sup` and `inf`,
  -- and separates points strongly (since `A` does),
  -- so we can apply `sublattice_closure_eq_top`.
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w : A.SeparatesPoints
    ⊢ Eq A.topologicalClosure Top.top
  -/
  apply SetLike.ext'
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w : A.SeparatesPoints
    ⊢ Eq ↑A.topologicalClosure ↑Top.top
  -/
  let L := A.topologicalClosure
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w : A.SeparatesPoints
    L : Subalgebra Real (ContinuousMap X Real) := A.topologicalClosure
    ⊢ Eq ↑A.topologicalClosure ↑Top.top
  -/
  have n : Set.Nonempty (L : Set C(X, ℝ)) := ⟨(1 : C(X, ℝ)), A.le_topologicalClosure A.one_mem⟩
  convert
    sublattice_closure_eq_top (L : Set C(X, ℝ)) n
      (fun f fm g gm => inf_mem_closed_subalgebra L A.isClosed_topologicalClosure ⟨f, fm⟩ ⟨g, gm⟩)
      (fun f fm g gm => sup_mem_closed_subalgebra L A.isClosed_topologicalClosure ⟨f, fm⟩ ⟨g, gm⟩)
      (Subalgebra.SeparatesPoints.strongly
        (Subalgebra.separatesPoints_monotone A.le_topologicalClosure w))
  /-
    case h.e'_2
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w : A.SeparatesPoints
    L : Subalgebra Real (ContinuousMap X Real) := A.topologicalClosure
    n : (↑L).Nonempty
    ⊢ Eq (↑A.topologicalClosure) (closure ↑L)
  -/
  simp [L]
  /-
    🎉 no goals
  -/


/-- An alternative statement of the Stone-Weierstrass theorem.

If `A` is a subalgebra of `C(X, ℝ)` which separates points (and `X` is compact),
every real-valued continuous function on `X` is a uniform limit of elements of `A`.
-/
theorem continuousMap_mem_subalgebra_closure_of_separatesPoints (A : Subalgebra ℝ C(X, ℝ))
    (w : A.SeparatesPoints) (f : C(X, ℝ)) : f ∈ A.topologicalClosure := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w : A.SeparatesPoints
    f : ContinuousMap X Real
    ⊢ Membership.mem A.topologicalClosure f
  -/
  rw [subalgebra_topologicalClosure_eq_top_of_separatesPoints A w]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w : A.SeparatesPoints
    f : ContinuousMap X Real
    ⊢ Membership.mem Top.top f
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An alternative statement of the Stone-Weierstrass theorem,
for those who like their epsilons.

If `A` is a subalgebra of `C(X, ℝ)` which separates points (and `X` is compact),
every real-valued continuous function on `X` is within any `ε > 0` of some element of `A`.
-/
theorem exists_mem_subalgebra_near_continuousMap_of_separatesPoints (A : Subalgebra ℝ C(X, ℝ))
    (w : A.SeparatesPoints) (f : C(X, ℝ)) (ε : ℝ) (pos : 0 < ε) :
    ∃ g : A, ‖(g : C(X, ℝ)) - f‖ < ε := by
  have w :=
    mem_closure_iff_frequently.mp (continuousMap_mem_subalgebra_closure_of_separatesPoints A w f)
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w✝ : A.SeparatesPoints
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    w : Filter.Frequently (fun x => Membership.mem (↑A.toSubsemiring) x) (nhds f)
    ⊢ Exists fun g => LT.lt (Norm.norm (HSub.hSub (↑g) f)) ε
  -/
  rw [Metric.nhds_basis_ball.frequently_iff] at w
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w✝ : A.SeparatesPoints
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    w : ∀ (i : Real), LT.lt 0 i → Exists fun x => And (Membership.mem (Metric.ball …
    ⊢ Exists fun g => LT.lt (Norm.norm (HSub.hSub (↑g) f)) ε
  -/
  obtain ⟨g, H, m⟩ := w ε pos
  /-
    case intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w✝ : A.SeparatesPoints
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    w : ∀ (i : Real), LT.lt 0 i → Exists fun x => And (Membership.mem (Metric.ball …
    g : ContinuousMap X Real
    H : Membership.mem (Metric.ball f ε) g
    m : Membership.mem (↑A.toSubsemiring) g
    ⊢ Exists fun g => LT.lt (Norm.norm (HSub.hSub (↑g) f)) ε
  -/
  rw [Metric.mem_ball, dist_eq_norm] at H
  /-
    case intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w✝ : A.SeparatesPoints
    f : ContinuousMap X Real
    ε : Real
    pos : LT.lt 0 ε
    w : ∀ (i : Real), LT.lt 0 i → Exists fun x => And (Membership.mem (Metric.ball …
    g : ContinuousMap X Real
    H : LT.lt (Norm.norm (HSub.hSub g f)) ε
    m : Membership.mem (↑A.toSubsemiring) g
    ⊢ Exists fun g => LT.lt (Norm.norm (HSub.hSub (↑g) f)) ε
  -/
  exact ⟨⟨g, m⟩, H⟩
  /-
    🎉 no goals
  -/


/-- An alternative statement of the Stone-Weierstrass theorem,
for those who like their epsilons and don't like bundled continuous functions.

If `A` is a subalgebra of `C(X, ℝ)` which separates points (and `X` is compact),
every real-valued continuous function on `X` is within any `ε > 0` of some element of `A`.
-/
theorem exists_mem_subalgebra_near_continuous_of_separatesPoints (A : Subalgebra ℝ C(X, ℝ))
    (w : A.SeparatesPoints) (f : X → ℝ) (c : Continuous f) (ε : ℝ) (pos : 0 < ε) :
    ∃ g : A, ∀ x, ‖(g : X → ℝ) x - f x‖ < ε := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w : A.SeparatesPoints
    f : X → Real
    c : Continuous f
    ε : Real
    pos : LT.lt 0 ε
    ⊢ Exists fun g => ∀ (x : X), LT.lt (Norm.norm (HSub.hSub (↑g x) (f x))) ε
  -/
  obtain ⟨g, b⟩ := exists_mem_subalgebra_near_continuousMap_of_separatesPoints A w ⟨f, c⟩ ε pos
  /-
    case intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w : A.SeparatesPoints
    f : X → Real
    c : Continuous f
    ε : Real
    pos : LT.lt 0 ε
    g : Subtype fun x => Membership.mem A x
    b : LT.lt (Norm.norm (HSub.hSub ↑g { toFun := f, continuous_toFun := c })) ε
    ⊢ Exists fun g => ∀ (x : X), LT.lt (Norm.norm (HSub.hSub (↑g x) (f x))) ε
  -/
  use g
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : Subalgebra Real (ContinuousMap X Real)
    w : A.SeparatesPoints
    f : X → Real
    c : Continuous f
    ε : Real
    pos : LT.lt 0 ε
    g : Subtype fun x => Membership.mem A x
    b : LT.lt (Norm.norm (HSub.hSub ↑g { toFun := f, continuous_toFun := c })) ε
    ⊢ ∀ (x : X), LT.lt (Norm.norm (HSub.hSub (↑g x) (f x))) ε
  -/
  rwa [norm_lt_iff _ pos] at b
  /-
    🎉 no goals
  -/


/-- If a star subalgebra of `C(X, 𝕜)` separates points, then the real subalgebra
of its purely real-valued elements also separates points. -/
theorem Subalgebra.SeparatesPoints.rclike_to_real {A : StarSubalgebra 𝕜 C(X, 𝕜)}
    (hA : A.SeparatesPoints) :
      ((A.restrictScalars ℝ).comap
        (ofRealAm.compLeftContinuous ℝ continuous_ofReal)).SeparatesPoints := by
  /-
    𝕜 : Type u_1
    X : Type u_2
    inst✝¹ : RCLike 𝕜
    inst✝ : TopologicalSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    ⊢ (Subalgebra.comap (AlgHom.compLeftContinuous Real RCLike.ofRealAm ⋯) (Subalg …
  -/
  intro x₁ x₂ hx
  -- Let `f` in the subalgebra `A` separate the points `x₁`, `x₂`
  /-
    𝕜 : Type u_1
    X : Type u_2
    inst✝¹ : RCLike 𝕜
    inst✝ : TopologicalSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    x₁ x₂ : X
    hx : Ne x₁ x₂
    ⊢ Exists fun f => And (Membership.mem (Set.image (fun f => ⇑f) ↑(Subalgebra.co …
  -/
  obtain ⟨_, ⟨f, hfA, rfl⟩, hf⟩ := hA hx
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    X : Type u_2
    inst✝¹ : RCLike 𝕜
    inst✝ : TopologicalSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    x₁ x₂ : X
    hx : Ne x₁ x₂
    f : ContinuousMap X 𝕜
    hfA : Membership.mem (↑A.toSubalgebra) f
    hf : Ne ((fun f => ⇑f) f x₁) ((fun f => ⇑f) f x₂)
    ⊢ Exists fun f => And (Membership.mem (Set.image (fun f => ⇑f) ↑(Subalgebra.co …
  -/
  let F : C(X, 𝕜) := f - const _ (f x₂)
  -- Subtract the constant `f x₂` from `f`; this is still an element of the subalgebra
  have hFA : F ∈ A := by
    refine A.sub_mem hfA (@Eq.subst _ (· ∈ A) _ _ ?_ <| A.smul_mem A.one_mem <| f x₂)
    ext1
    simp only [coe_smul, coe_one, smul_apply, one_apply, Algebra.id.smul_eq_mul, mul_one,
      const_apply]
  -- Consider now the function `fun x ↦ |f x - f x₂| ^ 2`
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    X : Type u_2
    inst✝¹ : RCLike 𝕜
    inst✝ : TopologicalSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    x₁ x₂ : X
    hx : Ne x₁ x₂
    f : ContinuousMap X 𝕜
    hfA : Membership.mem (↑A.toSubalgebra) f
    hf : Ne ((fun f => ⇑f) f x₁) ((fun f => ⇑f) f x₂)
    F : ContinuousMap X 𝕜 := HSub.hSub f (ContinuousMap.const X (f x₂))
    hFA : Membership.mem A F
    ⊢ Exists fun f => And (Membership.mem (Set.image (fun f => ⇑f) ↑(Subalgebra.co …
  -/
  refine ⟨_, ⟨⟨(‖F ·‖ ^ 2), by continuity⟩, ?_, rfl⟩, ?_⟩
  · -- This is also an element of the subalgebra, and takes only real values
    /-
      case intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      X : Type u_2
      inst✝¹ : RCLike 𝕜
      inst✝ : TopologicalSpace X
      A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
      hA : A.SeparatesPoints
      x₁ x₂ : X
      hx : Ne x₁ x₂
      f : ContinuousMap X 𝕜
      hfA : Membership.mem (↑A.toSubalgebra) f
      hf : Ne ((fun f => ⇑f) f x₁) ((fun f => ⇑f) f x₂)
      F : ContinuousMap X 𝕜 := HSub.hSub f (ContinuousMap.const X (f x₂))
      hFA : Membership.mem A F
      ⊢ Membership.mem ↑(Subalgebra.comap (AlgHom.compLeftContinuous Real RCLike.ofR …
    -/
    rw [SetLike.mem_coe, Subalgebra.mem_comap]
    /-
      case intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      X : Type u_2
      inst✝¹ : RCLike 𝕜
      inst✝ : TopologicalSpace X
      A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
      hA : A.SeparatesPoints
      x₁ x₂ : X
      hx : Ne x₁ x₂
      f : ContinuousMap X 𝕜
      hfA : Membership.mem (↑A.toSubalgebra) f
      hf : Ne ((fun f => ⇑f) f x₁) ((fun f => ⇑f) f x₂)
      F : ContinuousMap X 𝕜 := HSub.hSub f (ContinuousMap.const X (f x₂))
      hFA : Membership.mem A F
      ⊢ Membership.mem (Subalgebra.restrictScalars Real A.toSubalgebra) ((AlgHom.com …
    -/
    convert (A.restrictScalars ℝ).mul_mem hFA (star_mem hFA : star F ∈ A)
    /-
      case h.e'_5
      𝕜 : Type u_1
      X : Type u_2
      inst✝¹ : RCLike 𝕜
      inst✝ : TopologicalSpace X
      A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
      hA : A.SeparatesPoints
      x₁ x₂ : X
      hx : Ne x₁ x₂
      f : ContinuousMap X 𝕜
      hfA : Membership.mem (↑A.toSubalgebra) f
      hf : Ne ((fun f => ⇑f) f x₁) ((fun f => ⇑f) f x₂)
      F : ContinuousMap X 𝕜 := HSub.hSub f (ContinuousMap.const X (f x₂))
      hFA : Membership.mem A F
      ⊢ Eq ((AlgHom.compLeftContinuous Real RCLike.ofRealAm ⋯) { toFun := fun x => H …
    -/
    ext1
    /-
      case h.e'_5.h
      𝕜 : Type u_1
      X : Type u_2
      inst✝¹ : RCLike 𝕜
      inst✝ : TopologicalSpace X
      A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
      hA : A.SeparatesPoints
      x₁ x₂ : X
      hx : Ne x₁ x₂
      f : ContinuousMap X 𝕜
      hfA : Membership.mem (↑A.toSubalgebra) f
      hf : Ne ((fun f => ⇑f) f x₁) ((fun f => ⇑f) f x₂)
      F : ContinuousMap X 𝕜 := HSub.hSub f (ContinuousMap.const X (f x₂))
      hFA : Membership.mem A F
      a✝ : X
      ⊢ Eq (((AlgHom.compLeftContinuous Real RCLike.ofRealAm ⋯) { toFun := fun x =>  …
    -/
    simp [← RCLike.mul_conj]
    /-
      🎉 no goals
    -/
  · -- And it also separates the points `x₁`, `x₂`
    /-
      case intro.intro.intro.intro.refine_2
      𝕜 : Type u_1
      X : Type u_2
      inst✝¹ : RCLike 𝕜
      inst✝ : TopologicalSpace X
      A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
      hA : A.SeparatesPoints
      x₁ x₂ : X
      hx : Ne x₁ x₂
      f : ContinuousMap X 𝕜
      hfA : Membership.mem (↑A.toSubalgebra) f
      hf : Ne ((fun f => ⇑f) f x₁) ((fun f => ⇑f) f x₂)
      F : ContinuousMap X 𝕜 := HSub.hSub f (ContinuousMap.const X (f x₂))
      hFA : Membership.mem A F
      ⊢ Ne ((fun f => ⇑f) { toFun := fun x => HPow.hPow (Norm.norm (F x)) 2, continu …
    -/
    simpa [F] using sub_ne_zero.mpr hf
    /-
      🎉 no goals
    -/


/-- The Stone-Weierstrass approximation theorem, `RCLike` version, that a star subalgebra `A` of
`C(X, 𝕜)`, where `X` is a compact topological space and `RCLike 𝕜`, is dense if it separates
points. -/
theorem ContinuousMap.starSubalgebra_topologicalClosure_eq_top_of_separatesPoints
    (A : StarSubalgebra 𝕜 C(X, 𝕜)) (hA : A.SeparatesPoints) : A.topologicalClosure = ⊤ := by
  /-
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    ⊢ Eq A.topologicalClosure Top.top
  -/
  rw [StarSubalgebra.eq_top_iff]
  -- Let `I` be the natural inclusion of `C(X, ℝ)` into `C(X, 𝕜)`
  /-
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    ⊢ ∀ (x : ContinuousMap X 𝕜), Membership.mem A.topologicalClosure x
  -/
  let I : C(X, ℝ) →ₗ[ℝ] C(X, 𝕜) := ofRealCLM.compLeftContinuous ℝ X
  -- The main point of the proof is that its range (i.e., every real-valued function) is contained
  -- in the closure of `A`
  have key : LinearMap.range I ≤ (A.toSubmodule.restrictScalars ℝ).topologicalClosure := by
    -- Let `A₀` be the subalgebra of `C(X, ℝ)` consisting of `A`'s purely real elements; it is the
    -- preimage of `A` under `I`.  In this argument we only need its submodule structure.
    let A₀ : Submodule ℝ C(X, ℝ) := (A.toSubmodule.restrictScalars ℝ).comap I
    -- By `Subalgebra.SeparatesPoints.rclike_to_real`, this subalgebra also separates points, so
    -- we may apply the real Stone-Weierstrass result to it.
    have SW : A₀.topologicalClosure = ⊤ :=
      haveI := subalgebra_topologicalClosure_eq_top_of_separatesPoints _ hA.rclike_to_real
      congr_arg Subalgebra.toSubmodule this
    rw [← Submodule.map_top, ← SW]
    -- So it suffices to prove that the image under `I` of the closure of `A₀` is contained in the
    -- closure of `A`, which follows by abstract nonsense
    have h₁ := A₀.topologicalClosure_map ((@ofRealCLM 𝕜 _).compLeftContinuousCompact X)
    have h₂ := (A.toSubmodule.restrictScalars ℝ).map_comap_le I
    exact h₁.trans (Submodule.topologicalClosure_mono h₂)
  -- In particular, for a function `f` in `C(X, 𝕜)`, the real and imaginary parts of `f` are in the
  -- closure of `A`
  /-
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    I : LinearMap (RingHom.id Real) (ContinuousMap X Real) (ContinuousMap X 𝕜) :=  …
    key : LE.le (LinearMap.range I) (Submodule.restrictScalars Real (Subalgebra.to …
    ⊢ ∀ (x : ContinuousMap X 𝕜), Membership.mem A.topologicalClosure x
  -/
  intro f
  /-
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    I : LinearMap (RingHom.id Real) (ContinuousMap X Real) (ContinuousMap X 𝕜) :=  …
    key : LE.le (LinearMap.range I) (Submodule.restrictScalars Real (Subalgebra.to …
    f : ContinuousMap X 𝕜
    ⊢ Membership.mem A.topologicalClosure f
  -/
  let f_re : C(X, ℝ) := (⟨RCLike.re, RCLike.reCLM.continuous⟩ : C(𝕜, ℝ)).comp f
  /-
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    I : LinearMap (RingHom.id Real) (ContinuousMap X Real) (ContinuousMap X 𝕜) :=  …
    key : LE.le (LinearMap.range I) (Submodule.restrictScalars Real (Subalgebra.to …
    f : ContinuousMap X 𝕜
    f_re : ContinuousMap X Real := { toFun := ⇑RCLike.re, continuous_toFun := ⋯ }. …
    ⊢ Membership.mem A.topologicalClosure f
  -/
  let f_im : C(X, ℝ) := (⟨RCLike.im, RCLike.imCLM.continuous⟩ : C(𝕜, ℝ)).comp f
  /-
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    I : LinearMap (RingHom.id Real) (ContinuousMap X Real) (ContinuousMap X 𝕜) :=  …
    key : LE.le (LinearMap.range I) (Submodule.restrictScalars Real (Subalgebra.to …
    f : ContinuousMap X 𝕜
    f_re : ContinuousMap X Real := { toFun := ⇑RCLike.re, continuous_toFun := ⋯ }. …
    f_im : ContinuousMap X Real := { toFun := ⇑RCLike.im, continuous_toFun := ⋯ }. …
    ⊢ Membership.mem A.topologicalClosure f
  -/
  have h_f_re : I f_re ∈ A.topologicalClosure := key ⟨f_re, rfl⟩
  /-
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    I : LinearMap (RingHom.id Real) (ContinuousMap X Real) (ContinuousMap X 𝕜) :=  …
    key : LE.le (LinearMap.range I) (Submodule.restrictScalars Real (Subalgebra.to …
    f : ContinuousMap X 𝕜
    f_re : ContinuousMap X Real := { toFun := ⇑RCLike.re, continuous_toFun := ⋯ }. …
    f_im : ContinuousMap X Real := { toFun := ⇑RCLike.im, continuous_toFun := ⋯ }. …
    h_f_re : Membership.mem A.topologicalClosure (I f_re)
    ⊢ Membership.mem A.topologicalClosure f
  -/
  have h_f_im : I f_im ∈ A.topologicalClosure := key ⟨f_im, rfl⟩
  -- So `f_re + I • f_im` is in the closure of `A`
  /-
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    I : LinearMap (RingHom.id Real) (ContinuousMap X Real) (ContinuousMap X 𝕜) :=  …
    key : LE.le (LinearMap.range I) (Submodule.restrictScalars Real (Subalgebra.to …
    f : ContinuousMap X 𝕜
    f_re : ContinuousMap X Real := { toFun := ⇑RCLike.re, continuous_toFun := ⋯ }. …
    f_im : ContinuousMap X Real := { toFun := ⇑RCLike.im, continuous_toFun := ⋯ }. …
    h_f_re : Membership.mem A.topologicalClosure (I f_re)
    h_f_im : Membership.mem A.topologicalClosure (I f_im)
    ⊢ Membership.mem A.topologicalClosure f
  -/
  have := A.topologicalClosure.add_mem h_f_re (A.topologicalClosure.smul_mem h_f_im RCLike.I)
  /-
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    I : LinearMap (RingHom.id Real) (ContinuousMap X Real) (ContinuousMap X 𝕜) :=  …
    key : LE.le (LinearMap.range I) (Submodule.restrictScalars Real (Subalgebra.to …
    f : ContinuousMap X 𝕜
    f_re : ContinuousMap X Real := { toFun := ⇑RCLike.re, continuous_toFun := ⋯ }. …
    f_im : ContinuousMap X Real := { toFun := ⇑RCLike.im, continuous_toFun := ⋯ }. …
    h_f_re : Membership.mem A.topologicalClosure (I f_re)
    h_f_im : Membership.mem A.topologicalClosure (I f_im)
    this : Membership.mem A.topologicalClosure.toSubalgebra (HAdd.hAdd (I f_re) (H …
    ⊢ Membership.mem A.topologicalClosure f
  -/
  rw [StarSubalgebra.mem_toSubalgebra] at this
  /-
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    I : LinearMap (RingHom.id Real) (ContinuousMap X Real) (ContinuousMap X 𝕜) :=  …
    key : LE.le (LinearMap.range I) (Submodule.restrictScalars Real (Subalgebra.to …
    f : ContinuousMap X 𝕜
    f_re : ContinuousMap X Real := { toFun := ⇑RCLike.re, continuous_toFun := ⋯ }. …
    f_im : ContinuousMap X Real := { toFun := ⇑RCLike.im, continuous_toFun := ⋯ }. …
    h_f_re : Membership.mem A.topologicalClosure (I f_re)
    h_f_im : Membership.mem A.topologicalClosure (I f_im)
    this : Membership.mem A.topologicalClosure (HAdd.hAdd (I f_re) (HSMul.hSMul RC …
    ⊢ Membership.mem A.topologicalClosure f
  -/
  convert this
  -- And this, of course, is just `f`
  /-
    case h.e'_5
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    I : LinearMap (RingHom.id Real) (ContinuousMap X Real) (ContinuousMap X 𝕜) :=  …
    key : LE.le (LinearMap.range I) (Submodule.restrictScalars Real (Subalgebra.to …
    f : ContinuousMap X 𝕜
    f_re : ContinuousMap X Real := { toFun := ⇑RCLike.re, continuous_toFun := ⋯ }. …
    f_im : ContinuousMap X Real := { toFun := ⇑RCLike.im, continuous_toFun := ⋯ }. …
    h_f_re : Membership.mem A.topologicalClosure (I f_re)
    h_f_im : Membership.mem A.topologicalClosure (I f_im)
    this : Membership.mem A.topologicalClosure (HAdd.hAdd (I f_re) (HSMul.hSMul RC …
    ⊢ Eq f (HAdd.hAdd (I f_re) (HSMul.hSMul RCLike.I (I f_im)))
  -/
  ext
  /-
    case h.e'_5.h
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    I : LinearMap (RingHom.id Real) (ContinuousMap X Real) (ContinuousMap X 𝕜) :=  …
    key : LE.le (LinearMap.range I) (Submodule.restrictScalars Real (Subalgebra.to …
    f : ContinuousMap X 𝕜
    f_re : ContinuousMap X Real := { toFun := ⇑RCLike.re, continuous_toFun := ⋯ }. …
    f_im : ContinuousMap X Real := { toFun := ⇑RCLike.im, continuous_toFun := ⋯ }. …
    h_f_re : Membership.mem A.topologicalClosure (I f_re)
    h_f_im : Membership.mem A.topologicalClosure (I f_im)
    this : Membership.mem A.topologicalClosure (HAdd.hAdd (I f_re) (HSMul.hSMul RC …
    a✝ : X
    ⊢ Eq (f a✝) ((HAdd.hAdd (I f_re) (HSMul.hSMul RCLike.I (I f_im))) a✝)
  -/
  apply Eq.symm
  /-
    case h.e'_5.h.h
    𝕜 : Type u_1
    X : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    A : StarSubalgebra 𝕜 (ContinuousMap X 𝕜)
    hA : A.SeparatesPoints
    I : LinearMap (RingHom.id Real) (ContinuousMap X Real) (ContinuousMap X 𝕜) :=  …
    key : LE.le (LinearMap.range I) (Submodule.restrictScalars Real (Subalgebra.to …
    f : ContinuousMap X 𝕜
    f_re : ContinuousMap X Real := { toFun := ⇑RCLike.re, continuous_toFun := ⋯ }. …
    f_im : ContinuousMap X Real := { toFun := ⇑RCLike.im, continuous_toFun := ⋯ }. …
    h_f_re : Membership.mem A.topologicalClosure (I f_re)
    h_f_im : Membership.mem A.topologicalClosure (I f_im)
    this : Membership.mem A.topologicalClosure (HAdd.hAdd (I f_re) (HSMul.hSMul RC …
    a✝ : X
    ⊢ Eq ((HAdd.hAdd (I f_re) (HSMul.hSMul RCLike.I (I f_im))) a✝) (f a✝)
  -/
  simp [I, f_re, f_im, mul_comm RCLike.I _]
  /-
    🎉 no goals
  -/


/-- Polynomial functions in are dense in `C(s, ℝ)` when `s` is compact.

See `polynomialFunctions_closure_eq_top` for the special case `s = Set.Icc a b` which does not use
the full Stone-Weierstrass theorem. Of course, that version could be used to prove this one as
well. -/
theorem polynomialFunctions.topologicalClosure (s : Set ℝ)
    [CompactSpace s] : (polynomialFunctions s).topologicalClosure = ⊤ :=
  ContinuousMap.subalgebra_topologicalClosure_eq_top_of_separatesPoints _
    (polynomialFunctions_separatesPoints s)


/-- The star subalgebra generated by polynomials functions is dense in `C(s, 𝕜)` when `s` is
compact and `𝕜` is either `ℝ` or `ℂ`. -/
theorem polynomialFunctions.starClosure_topologicalClosure {𝕜 : Type*} [RCLike 𝕜] (s : Set 𝕜)
    [CompactSpace s] : (polynomialFunctions s).starClosure.topologicalClosure = ⊤ :=
  ContinuousMap.starSubalgebra_topologicalClosure_eq_top_of_separatesPoints _
    (Subalgebra.separatesPoints_monotone le_sup_left (polynomialFunctions_separatesPoints s))


/-- An induction principle for `C(s, 𝕜)`. -/
@[elab_as_elim]
theorem ContinuousMap.induction_on {𝕜 : Type*} [RCLike 𝕜] {s : Set 𝕜}
    {p : C(s, 𝕜) → Prop} (const : ∀ r, p (.const s r)) (id : p (.restrict s <| .id 𝕜))
    (star_id : p (star (.restrict s <| .id 𝕜)))
    (add : ∀ f g, p f → p g → p (f + g)) (mul : ∀ f g, p f → p g → p (f * g))
    (closure : (∀ f ∈ (polynomialFunctions s).starClosure, p f) → ∀ f, p f) (f : C(s, 𝕜)) :
    p f := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    s : Set 𝕜
    p : ContinuousMap (↑s) 𝕜 → Prop
    const : ∀ (r : 𝕜), p (ContinuousMap.const (↑s) r)
    id : p (ContinuousMap.restrict s (ContinuousMap.id 𝕜))
    star_id : p (Star.star (ContinuousMap.restrict s (ContinuousMap.id 𝕜)))
    add : ∀ (f g : ContinuousMap (↑s) 𝕜), p f → p g → p (HAdd.hAdd f g)
    mul : ∀ (f g : ContinuousMap (↑s) 𝕜), p f → p g → p (HMul.hMul f g)
    closure : (∀ (f : ContinuousMap (↑s) 𝕜), Membership.mem (polynomialFunctions s …
    f : ContinuousMap (↑s) 𝕜
    ⊢ p f
  -/
  refine closure (fun f hf => ?_) f
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    s : Set 𝕜
    p : ContinuousMap (↑s) 𝕜 → Prop
    const : ∀ (r : 𝕜), p (ContinuousMap.const (↑s) r)
    id : p (ContinuousMap.restrict s (ContinuousMap.id 𝕜))
    star_id : p (Star.star (ContinuousMap.restrict s (ContinuousMap.id 𝕜)))
    add : ∀ (f g : ContinuousMap (↑s) 𝕜), p f → p g → p (HAdd.hAdd f g)
    mul : ∀ (f g : ContinuousMap (↑s) 𝕜), p f → p g → p (HMul.hMul f g)
    closure : (∀ (f : ContinuousMap (↑s) 𝕜), Membership.mem (polynomialFunctions s …
    f✝ f : ContinuousMap (↑s) 𝕜
    hf : Membership.mem (polynomialFunctions s).starClosure f
    ⊢ p f
  -/
  rw [polynomialFunctions.starClosure_eq_adjoin_X] at hf
  induction hf using Algebra.adjoin_induction with
  | mem f hf =>
    simp only [Set.mem_union, Set.mem_singleton_iff, Set.mem_star] at hf
    rw [star_eq_iff_star_eq, eq_comm (b := f)] at hf
    obtain (rfl | rfl) := hf
    all_goals simpa only [toContinuousMapOnAlgHom_apply, toContinuousMapOn_X_eq_restrict_id]
  | algebraMap r => exact const r
  | add _ _ _ _ hf hg => exact add _ _ hf hg
  | mul _ _ _ _ hf hg => exact mul _ _ hf hg


open Topology in
@[elab_as_elim]
theorem ContinuousMap.induction_on_of_compact {𝕜 : Type*} [RCLike 𝕜] {s : Set 𝕜} [CompactSpace s]
    {p : C(s, 𝕜) → Prop} (const : ∀ r, p (.const s r)) (id : p (.restrict s <| .id 𝕜))
    (star_id : p (star (.restrict s <| .id 𝕜)))
    (add : ∀ f g, p f → p g → p (f + g)) (mul : ∀ f g, p f → p g → p (f * g))
    (frequently : ∀ f, (∃ᶠ g in 𝓝 f, p g) → p f) (f : C(s, 𝕜)) :
    p f := by
  /-
    𝕜 : Type u_1
    inst✝¹ : RCLike 𝕜
    s : Set 𝕜
    inst✝ : CompactSpace ↑s
    p : ContinuousMap (↑s) 𝕜 → Prop
    const : ∀ (r : 𝕜), p (ContinuousMap.const (↑s) r)
    id : p (ContinuousMap.restrict s (ContinuousMap.id 𝕜))
    star_id : p (Star.star (ContinuousMap.restrict s (ContinuousMap.id 𝕜)))
    add : ∀ (f g : ContinuousMap (↑s) 𝕜), p f → p g → p (HAdd.hAdd f g)
    mul : ∀ (f g : ContinuousMap (↑s) 𝕜), p f → p g → p (HMul.hMul f g)
    frequently : ∀ (f : ContinuousMap (↑s) 𝕜), Filter.Frequently (fun g => p g) (n …
    f : ContinuousMap (↑s) 𝕜
    ⊢ p f
  -/
  refine f.induction_on const id star_id add mul fun h f ↦ frequently f ?_
  /-
    𝕜 : Type u_1
    inst✝¹ : RCLike 𝕜
    s : Set 𝕜
    inst✝ : CompactSpace ↑s
    p : ContinuousMap (↑s) 𝕜 → Prop
    const : ∀ (r : 𝕜), p (ContinuousMap.const (↑s) r)
    id : p (ContinuousMap.restrict s (ContinuousMap.id 𝕜))
    star_id : p (Star.star (ContinuousMap.restrict s (ContinuousMap.id 𝕜)))
    add : ∀ (f g : ContinuousMap (↑s) 𝕜), p f → p g → p (HAdd.hAdd f g)
    mul : ∀ (f g : ContinuousMap (↑s) 𝕜), p f → p g → p (HMul.hMul f g)
    frequently : ∀ (f : ContinuousMap (↑s) 𝕜), Filter.Frequently (fun g => p g) (n …
    f✝ : ContinuousMap (↑s) 𝕜
    h : ∀ (f : ContinuousMap (↑s) 𝕜), Membership.mem (polynomialFunctions s).starC …
    f : ContinuousMap (↑s) 𝕜
    ⊢ Filter.Frequently (fun g => p g) (nhds f)
  -/
  have := polynomialFunctions.starClosure_topologicalClosure s ▸ mem_top (x := f)
  /-
    𝕜 : Type u_1
    inst✝¹ : RCLike 𝕜
    s : Set 𝕜
    inst✝ : CompactSpace ↑s
    p : ContinuousMap (↑s) 𝕜 → Prop
    const : ∀ (r : 𝕜), p (ContinuousMap.const (↑s) r)
    id : p (ContinuousMap.restrict s (ContinuousMap.id 𝕜))
    star_id : p (Star.star (ContinuousMap.restrict s (ContinuousMap.id 𝕜)))
    add : ∀ (f g : ContinuousMap (↑s) 𝕜), p f → p g → p (HAdd.hAdd f g)
    mul : ∀ (f g : ContinuousMap (↑s) 𝕜), p f → p g → p (HMul.hMul f g)
    frequently : ∀ (f : ContinuousMap (↑s) 𝕜), Filter.Frequently (fun g => p g) (n …
    f✝ : ContinuousMap (↑s) 𝕜
    h : ∀ (f : ContinuousMap (↑s) 𝕜), Membership.mem (polynomialFunctions s).starC …
    f : ContinuousMap (↑s) 𝕜
    this : Membership.mem (polynomialFunctions s).starClosure.topologicalClosure f
    ⊢ Filter.Frequently (fun g => p g) (nhds f)
  -/
  rw [← SetLike.mem_coe, topologicalClosure_coe, mem_closure_iff_frequently] at this
  /-
    𝕜 : Type u_1
    inst✝¹ : RCLike 𝕜
    s : Set 𝕜
    inst✝ : CompactSpace ↑s
    p : ContinuousMap (↑s) 𝕜 → Prop
    const : ∀ (r : 𝕜), p (ContinuousMap.const (↑s) r)
    id : p (ContinuousMap.restrict s (ContinuousMap.id 𝕜))
    star_id : p (Star.star (ContinuousMap.restrict s (ContinuousMap.id 𝕜)))
    add : ∀ (f g : ContinuousMap (↑s) 𝕜), p f → p g → p (HAdd.hAdd f g)
    mul : ∀ (f g : ContinuousMap (↑s) 𝕜), p f → p g → p (HMul.hMul f g)
    frequently : ∀ (f : ContinuousMap (↑s) 𝕜), Filter.Frequently (fun g => p g) (n …
    f✝ : ContinuousMap (↑s) 𝕜
    h : ∀ (f : ContinuousMap (↑s) 𝕜), Membership.mem (polynomialFunctions s).starC …
    f : ContinuousMap (↑s) 𝕜
    this : Filter.Frequently (fun x => Membership.mem (↑(polynomialFunctions s).st …
    ⊢ Filter.Frequently (fun g => p g) (nhds f)
  -/
  exact this.mp <| .of_forall h
  /-
    🎉 no goals
  -/


/-- Continuous algebra homomorphisms from `C(s, ℝ)` into an `ℝ`-algebra `A` which agree
at `X : 𝕜[X]` (interpreted as a continuous map) are, in fact, equal. -/
@[ext (iff := false)]
theorem ContinuousMap.algHom_ext_map_X {A : Type*} [Ring A]
    [Algebra ℝ A] [TopologicalSpace A] [T2Space A] {s : Set ℝ} [CompactSpace s]
    {φ ψ : C(s, ℝ) →ₐ[ℝ] A} (hφ : Continuous φ) (hψ : Continuous ψ)
    (h : φ (toContinuousMapOnAlgHom s X) = ψ (toContinuousMapOnAlgHom s X)) : φ = ψ := by
  suffices (⊤ : Subalgebra ℝ C(s, ℝ)) ≤ AlgHom.equalizer φ ψ from
    AlgHom.ext fun x => this (by trivial)
  /-
    A : Type u_1
    inst✝⁴ : Ring A
    inst✝³ : Algebra Real A
    inst✝² : TopologicalSpace A
    inst✝¹ : T2Space A
    s : Set Real
    inst✝ : CompactSpace ↑s
    φ ψ : AlgHom Real (ContinuousMap (↑s) Real) A
    hφ : Continuous ⇑φ
    hψ : Continuous ⇑ψ
    h : Eq (φ ((Polynomial.toContinuousMapOnAlgHom s) Polynomial.X)) (ψ ((Polynomi …
    ⊢ LE.le Top.top (AlgHom.equalizer φ ψ)
  -/
  rw [← polynomialFunctions.topologicalClosure s]
  exact Subalgebra.topologicalClosure_minimal (polynomialFunctions s)
    (polynomialFunctions.le_equalizer s φ ψ h) (isClosed_eq hφ hψ)


/-- Continuous star algebra homomorphisms from `C(s, 𝕜)` into a star `𝕜`-algebra `A` which agree
at `X : 𝕜[X]` (interpreted as a continuous map) are, in fact, equal. -/
@[ext (iff := false)]
theorem ContinuousMap.starAlgHom_ext_map_X {𝕜 A : Type*} [RCLike 𝕜] [Ring A] [StarRing A]
    [Algebra 𝕜 A] [TopologicalSpace A] [T2Space A] {s : Set 𝕜} [CompactSpace s]
    {φ ψ : C(s, 𝕜) →⋆ₐ[𝕜] A} (hφ : Continuous φ) (hψ : Continuous ψ)
    (h : φ (toContinuousMapOnAlgHom s X) = ψ (toContinuousMapOnAlgHom s X)) : φ = ψ := by
  suffices (⊤ : StarSubalgebra 𝕜 C(s, 𝕜)) ≤ StarAlgHom.equalizer φ ψ from
    StarAlgHom.ext fun x => this mem_top
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : Ring A
    inst✝⁴ : StarRing A
    inst✝³ : Algebra 𝕜 A
    inst✝² : TopologicalSpace A
    inst✝¹ : T2Space A
    s : Set 𝕜
    inst✝ : CompactSpace ↑s
    φ ψ : StarAlgHom 𝕜 (ContinuousMap (↑s) 𝕜) A
    hφ : Continuous ⇑φ
    hψ : Continuous ⇑ψ
    h : Eq (φ ((Polynomial.toContinuousMapOnAlgHom s) Polynomial.X)) (ψ ((Polynomi …
    ⊢ LE.le Top.top (StarAlgHom.equalizer φ ψ)
  -/
  rw [← polynomialFunctions.starClosure_topologicalClosure s]
  exact StarSubalgebra.topologicalClosure_minimal
    (polynomialFunctions.starClosure_le_equalizer s φ ψ h) (isClosed_eq hφ hψ)


lemma adjoin_id_eq_span_one_union (s : Set 𝕜) :
    ((StarAlgebra.adjoin 𝕜 {(restrict s (.id 𝕜) : C(s, 𝕜))}) : Set C(s, 𝕜)) =
      span 𝕜 ({(1 : C(s, 𝕜))} ∪ (adjoin 𝕜 {(restrict s (.id 𝕜) : C(s, 𝕜))})) := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    s : Set 𝕜
    ⊢ Eq ↑(StarAlgebra.adjoin 𝕜 (Singleton.singleton (ContinuousMap.restrict s (Co …
  -/
  ext x
  rw [SetLike.mem_coe, SetLike.mem_coe, ← StarAlgebra.adjoin_nonUnitalStarSubalgebra,
    ← StarSubalgebra.mem_toSubalgebra, ← Subalgebra.mem_toSubmodule,
    StarAlgebra.adjoin_nonUnitalStarSubalgebra_eq_span, span_union, span_eq_toSubmodule]


open Pointwise in
lemma adjoin_id_eq_span_one_add (s : Set 𝕜) :
    ((StarAlgebra.adjoin 𝕜 {(restrict s (.id 𝕜) : C(s, 𝕜))}) : Set C(s, 𝕜)) =
      (span 𝕜 {(1 : C(s, 𝕜))} : Set C(s, 𝕜)) + (adjoin 𝕜 {(restrict s (.id 𝕜) : C(s, 𝕜))}) := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    s : Set 𝕜
    ⊢ Eq (↑(StarAlgebra.adjoin 𝕜 (Singleton.singleton (ContinuousMap.restrict s (C …
  -/
  ext x
  rw [SetLike.mem_coe, ← StarAlgebra.adjoin_nonUnitalStarSubalgebra,
    ← StarSubalgebra.mem_toSubalgebra, ← Subalgebra.mem_toSubmodule,
    StarAlgebra.adjoin_nonUnitalStarSubalgebra_eq_span, mem_sup]
  /-
    case h
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    s : Set 𝕜
    x : ContinuousMap (↑s) 𝕜
    ⊢ Iff (Exists fun y => And (Membership.mem (Submodule.span 𝕜 (Singleton.single …
  -/
  simp [Set.mem_add]
  /-
    🎉 no goals
  -/


lemma nonUnitalStarAlgebraAdjoin_id_subset_ker_evalStarAlgHom {s : Set 𝕜} (h0 : 0 ∈ s) :
    (adjoin 𝕜 {restrict s (.id 𝕜)} : Set C(s, 𝕜)) ⊆
      RingHom.ker (evalStarAlgHom 𝕜 𝕜 (⟨0, h0⟩ : s)) := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    s : Set 𝕜
    h0 : Membership.mem s 0
    ⊢ HasSubset.Subset ↑(NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Conti …
  -/
  intro f hf
  induction hf using adjoin_induction with
  | mem f hf =>
    obtain rfl := Set.mem_singleton_iff.mp hf
    rfl
  | add f g _ _ hf hg => exact add_mem hf hg
  | zero => exact zero_mem _
  | mul f g _ _ _ hg => exact Ideal.mul_mem_left _ f hg
  | smul r f _ hf =>
    rw [SetLike.mem_coe, RingHom.mem_ker] at hf ⊢
    rw [map_smul, hf, smul_zero]
  | star f _ hf =>
    rw [SetLike.mem_coe, RingHom.mem_ker] at hf ⊢
    rw [map_star, hf, star_zero]


lemma ker_evalStarAlgHom_inter_adjoin_id (s : Set 𝕜) (h0 : 0 ∈ s) :
    (StarAlgebra.adjoin 𝕜 {restrict s (.id 𝕜)} : Set C(s, 𝕜)) ∩
      RingHom.ker (evalStarAlgHom 𝕜 𝕜 (⟨0, h0⟩ : s)) = adjoin 𝕜 {restrict s (.id 𝕜)} := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    s : Set 𝕜
    h0 : Membership.mem s 0
    ⊢ Eq (Inter.inter ↑(StarAlgebra.adjoin 𝕜 (Singleton.singleton (ContinuousMap.r …
  -/
  ext f
  /-
    case h
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    s : Set 𝕜
    h0 : Membership.mem s 0
    f : ContinuousMap (↑s) 𝕜
    ⊢ Iff (Membership.mem (Inter.inter ↑(StarAlgebra.adjoin 𝕜 (Singleton.singleton …
  -/
  constructor
    /-
      case h.mp
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      s : Set 𝕜
      h0 : Membership.mem s 0
      f : ContinuousMap (↑s) 𝕜
      ⊢ Membership.mem (Inter.inter ↑(StarAlgebra.adjoin 𝕜 (Singleton.singleton (Con …
    -/
  · rintro ⟨hf₁, hf₂⟩
    /-
      case h.mp.intro
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      s : Set 𝕜
      h0 : Membership.mem s 0
      f : ContinuousMap (↑s) 𝕜
      hf₁ : Membership.mem (↑(StarAlgebra.adjoin 𝕜 (Singleton.singleton (ContinuousM …
      hf₂ : Membership.mem (↑(RingHom.ker (ContinuousMap.evalStarAlgHom 𝕜 𝕜 ⟨0, h0⟩) …
      ⊢ Membership.mem (↑(NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Contin …
    -/
    rw [SetLike.mem_coe] at hf₂ ⊢
    /-
      case h.mp.intro
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      s : Set 𝕜
      h0 : Membership.mem s 0
      f : ContinuousMap (↑s) 𝕜
      hf₁ : Membership.mem (↑(StarAlgebra.adjoin 𝕜 (Singleton.singleton (ContinuousM …
      hf₂ : Membership.mem (RingHom.ker (ContinuousMap.evalStarAlgHom 𝕜 𝕜 ⟨0, h0⟩)) f
      ⊢ Membership.mem (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Continuo …
    -/
    simp_rw [adjoin_id_eq_span_one_add, Set.mem_add, SetLike.mem_coe, mem_span_singleton] at hf₁
    /-
      case h.mp.intro
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      s : Set 𝕜
      h0 : Membership.mem s 0
      f : ContinuousMap (↑s) 𝕜
      hf₂ : Membership.mem (RingHom.ker (ContinuousMap.evalStarAlgHom 𝕜 𝕜 ⟨0, h0⟩)) f
      hf₁ : Exists fun x => And (Exists fun a => Eq (HSMul.hSMul a 1) x) (Exists fun …
      ⊢ Membership.mem (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Continuo …
    -/
    obtain ⟨-, ⟨r, rfl⟩, f, hf, rfl⟩ := hf₁
    /-
      case h.mp.intro.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      s : Set 𝕜
      h0 : Membership.mem s 0
      r : 𝕜
      f : ContinuousMap (↑s) 𝕜
      hf : Membership.mem (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Conti …
      hf₂ : Membership.mem (RingHom.ker (ContinuousMap.evalStarAlgHom 𝕜 𝕜 ⟨0, h0⟩))  …
      ⊢ Membership.mem (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Continuo …
    -/
    have := nonUnitalStarAlgebraAdjoin_id_subset_ker_evalStarAlgHom h0 hf
    /-
      case h.mp.intro.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      s : Set 𝕜
      h0 : Membership.mem s 0
      r : 𝕜
      f : ContinuousMap (↑s) 𝕜
      hf : Membership.mem (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Conti …
      hf₂ : Membership.mem (RingHom.ker (ContinuousMap.evalStarAlgHom 𝕜 𝕜 ⟨0, h0⟩))  …
      this : Membership.mem (↑(RingHom.ker (ContinuousMap.evalStarAlgHom 𝕜 𝕜 ⟨0, h0⟩ …
      ⊢ Membership.mem (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Continuo …
    -/
    simp only [SetLike.mem_coe, RingHom.mem_ker, evalStarAlgHom_apply] at hf₂ this
    /-
      case h.mp.intro.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      s : Set 𝕜
      h0 : Membership.mem s 0
      r : 𝕜
      f : ContinuousMap (↑s) 𝕜
      hf : Membership.mem (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Conti …
      hf₂ : Eq ((HAdd.hAdd (HSMul.hSMul r 1) f) ⟨0, h0⟩) 0
      this : Eq (f ⟨0, h0⟩) 0
      ⊢ Membership.mem (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Continuo …
    -/
    rw [add_apply, this, add_zero, smul_apply, one_apply, smul_eq_mul, mul_one] at hf₂
    /-
      case h.mp.intro.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      s : Set 𝕜
      h0 : Membership.mem s 0
      r : 𝕜
      f : ContinuousMap (↑s) 𝕜
      hf : Membership.mem (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Conti …
      hf₂ : Eq r 0
      this : Eq (f ⟨0, h0⟩) 0
      ⊢ Membership.mem (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Continuo …
    -/
    rwa [hf₂, zero_smul, zero_add]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      s : Set 𝕜
      h0 : Membership.mem s 0
      f : ContinuousMap (↑s) 𝕜
      ⊢ Membership.mem (↑(NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Contin …
    -/
  · simp only [Set.mem_inter_iff, SetLike.mem_coe]
    /-
      case h.mpr
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      s : Set 𝕜
      h0 : Membership.mem s 0
      f : ContinuousMap (↑s) 𝕜
      ⊢ Membership.mem (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Continuo …
    -/
    refine fun hf ↦ ⟨?_, nonUnitalStarAlgebraAdjoin_id_subset_ker_evalStarAlgHom h0 hf⟩
    /-
      case h.mpr
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      s : Set 𝕜
      h0 : Membership.mem s 0
      f : ContinuousMap (↑s) 𝕜
      hf : Membership.mem (NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Conti …
      ⊢ Membership.mem (StarAlgebra.adjoin 𝕜 (Singleton.singleton (ContinuousMap.res …
    -/
    exact adjoin_le_starAlgebra_adjoin _ _ hf
    /-
      🎉 no goals
    -/

-- the statement should be in terms of non unital subalgebras, but we lack API

open RingHom Filter Topology in
theorem AlgHom.closure_ker_inter {F S K A : Type*} [CommRing K] [Ring A] [Algebra K A]
    [TopologicalSpace K] [T1Space K] [TopologicalSpace A] [ContinuousSub A] [ContinuousSMul K A]
    [FunLike F A K] [AlgHomClass F K A K] [SetLike S A] [OneMemClass S A] [AddSubgroupClass S A]
    [SMulMemClass S K A] (φ : F) (hφ : Continuous φ) (s : S) :
    closure (s ∩ RingHom.ker φ) = closure s ∩ (ker φ : Set A) := by
  /-
    F : Type u_2
    S : Type u_3
    K : Type u_4
    A : Type u_5
    inst✝¹³ : CommRing K
    inst✝¹² : Ring A
    inst✝¹¹ : Algebra K A
    inst✝¹⁰ : TopologicalSpace K
    inst✝⁹ : T1Space K
    inst✝⁸ : TopologicalSpace A
    inst✝⁷ : ContinuousSub A
    inst✝⁶ : ContinuousSMul K A
    inst✝⁵ : FunLike F A K
    inst✝⁴ : AlgHomClass F K A K
    inst✝³ : SetLike S A
    inst✝² : OneMemClass S A
    inst✝¹ : AddSubgroupClass S A
    inst✝ : SMulMemClass S K A
    φ : F
    hφ : Continuous ⇑φ
    s : S
    ⊢ Eq (closure (Inter.inter ↑s ↑(RingHom.ker φ))) (Inter.inter (closure ↑s) ↑(R …
  -/
  refine subset_antisymm ?_ ?_
  · simpa only [ker_eq, (isClosed_singleton.preimage hφ).closure_eq]
      using closure_inter_subset_inter_closure s (ker φ : Set A)
    /-
      case refine_2
      F : Type u_2
      S : Type u_3
      K : Type u_4
      A : Type u_5
      inst✝¹³ : CommRing K
      inst✝¹² : Ring A
      inst✝¹¹ : Algebra K A
      inst✝¹⁰ : TopologicalSpace K
      inst✝⁹ : T1Space K
      inst✝⁸ : TopologicalSpace A
      inst✝⁷ : ContinuousSub A
      inst✝⁶ : ContinuousSMul K A
      inst✝⁵ : FunLike F A K
      inst✝⁴ : AlgHomClass F K A K
      inst✝³ : SetLike S A
      inst✝² : OneMemClass S A
      inst✝¹ : AddSubgroupClass S A
      inst✝ : SMulMemClass S K A
      φ : F
      hφ : Continuous ⇑φ
      s : S
      ⊢ HasSubset.Subset (Inter.inter (closure ↑s) ↑(RingHom.ker φ)) (closure (Inter …
    -/
  · intro x ⟨hxs, (hxφ : φ x = 0)⟩
    /-
      case refine_2
      F : Type u_2
      S : Type u_3
      K : Type u_4
      A : Type u_5
      inst✝¹³ : CommRing K
      inst✝¹² : Ring A
      inst✝¹¹ : Algebra K A
      inst✝¹⁰ : TopologicalSpace K
      inst✝⁹ : T1Space K
      inst✝⁸ : TopologicalSpace A
      inst✝⁷ : ContinuousSub A
      inst✝⁶ : ContinuousSMul K A
      inst✝⁵ : FunLike F A K
      inst✝⁴ : AlgHomClass F K A K
      inst✝³ : SetLike S A
      inst✝² : OneMemClass S A
      inst✝¹ : AddSubgroupClass S A
      inst✝ : SMulMemClass S K A
      φ : F
      hφ : Continuous ⇑φ
      s : S
      x : A
      hxs : Membership.mem (closure ↑s) x
      hxφ : Eq (φ x) 0
      ⊢ Membership.mem (closure (Inter.inter ↑s ↑(RingHom.ker φ))) x
    -/
    rw [mem_closure_iff_clusterPt, ClusterPt] at hxs
    have : Tendsto (fun y ↦ y - φ y • 1) (𝓝 x ⊓ 𝓟 s) (𝓝 x) := by
      conv => congr; rfl; rfl; rw [← sub_zero x, ← zero_smul K 1, ← hxφ]
      exact Filter.tendsto_inf_left (Continuous.tendsto (by fun_prop) x)
    /-
      case refine_2
      F : Type u_2
      S : Type u_3
      K : Type u_4
      A : Type u_5
      inst✝¹³ : CommRing K
      inst✝¹² : Ring A
      inst✝¹¹ : Algebra K A
      inst✝¹⁰ : TopologicalSpace K
      inst✝⁹ : T1Space K
      inst✝⁸ : TopologicalSpace A
      inst✝⁷ : ContinuousSub A
      inst✝⁶ : ContinuousSMul K A
      inst✝⁵ : FunLike F A K
      inst✝⁴ : AlgHomClass F K A K
      inst✝³ : SetLike S A
      inst✝² : OneMemClass S A
      inst✝¹ : AddSubgroupClass S A
      inst✝ : SMulMemClass S K A
      φ : F
      hφ : Continuous ⇑φ
      s : S
      x : A
      hxs : (Min.min (nhds x) (Filter.principal ↑s)).NeBot
      hxφ : Eq (φ x) 0
      this : Filter.Tendsto (fun y => HSub.hSub y (HSMul.hSMul (φ y) 1)) (Min.min (n …
      ⊢ Membership.mem (closure (Inter.inter ↑s ↑(RingHom.ker φ))) x
    -/
    refine mem_closure_of_tendsto this <| eventually_inf_principal.mpr ?_
    filter_upwards [] with g hg using
      ⟨sub_mem hg (SMulMemClass.smul_mem _ <| one_mem _), by simp [RingHom.mem_ker]⟩


lemma ker_evalStarAlgHom_eq_closure_adjoin_id (s : Set 𝕜) (h0 : 0 ∈ s) [CompactSpace s] :
    (RingHom.ker (evalStarAlgHom 𝕜 𝕜 (⟨0, h0⟩ : s)) : Set C(s, 𝕜)) =
      closure (adjoin 𝕜 {(restrict s (.id 𝕜))}) := by
  rw [← ker_evalStarAlgHom_inter_adjoin_id s h0,
    AlgHom.closure_ker_inter (φ := evalStarAlgHom 𝕜 𝕜 (X := s) ⟨0, h0⟩) (continuous_eval_const _) _]
  /-
    𝕜 : Type u_1
    inst✝¹ : RCLike 𝕜
    s : Set 𝕜
    h0 : Membership.mem s 0
    inst✝ : CompactSpace ↑s
    ⊢ Eq (↑(RingHom.ker (ContinuousMap.evalStarAlgHom 𝕜 𝕜 ⟨0, h0⟩))) (Inter.inter  …
  -/
  convert (Set.univ_inter _).symm
  rw [← Polynomial.toContinuousMapOn_X_eq_restrict_id, ← Polynomial.toContinuousMapOnAlgHom_apply,
    ← polynomialFunctions.starClosure_eq_adjoin_X s]
  /-
    case h.e'_3.h.e'_3
    𝕜 : Type u_1
    inst✝¹ : RCLike 𝕜
    s : Set 𝕜
    h0 : Membership.mem s 0
    inst✝ : CompactSpace ↑s
    ⊢ Eq (closure ↑(polynomialFunctions s).starClosure) Set.univ
  -/
  congrm(($(polynomialFunctions.starClosure_topologicalClosure s) : Set C(s, 𝕜)))
  /-
    🎉 no goals
  -/


/-- If `s : Set 𝕜` with `RCLike 𝕜` is compact and contains `0`, then the non-unital star subalgebra
generated by the identity function in `C(s, 𝕜)₀` is dense. This can be seen as a version of the
Weierstrass approximation theorem. -/
lemma ContinuousMapZero.adjoin_id_dense {s : Set 𝕜} [Zero s] (h0 : ((0 : s) : 𝕜) = 0)
    [CompactSpace s] : Dense (adjoin 𝕜 {(.id h0 : C(s, 𝕜)₀)} : Set C(s, 𝕜)₀) := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    s : Set 𝕜
    inst✝¹ : Zero ↑s
    h0 : Eq (↑0) 0
    inst✝ : CompactSpace ↑s
    ⊢ Dense ↑(NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (ContinuousMapZer …
  -/
  have h0' : 0 ∈ s := h0 ▸ (0 : s).property
  rw [dense_iff_closure_eq,
    ← isClosedEmbedding_toContinuousMap.injective.preimage_image (closure _),
    ← isClosedEmbedding_toContinuousMap.closure_image_eq, ← coe_toContinuousMapHom,
    ← NonUnitalStarSubalgebra.coe_map, NonUnitalStarAlgHom.map_adjoin_singleton,
    toContinuousMapHom_apply, toContinuousMap_id h0,
    ← ContinuousMap.ker_evalStarAlgHom_eq_closure_adjoin_id s h0']
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    s : Set 𝕜
    inst✝¹ : Zero ↑s
    h0 : Eq (↑0) 0
    inst✝ : CompactSpace ↑s
    h0' : Membership.mem s 0
    ⊢ Eq (Set.preimage ⇑ContinuousMapZero.toContinuousMapHom ↑(RingHom.ker (Contin …
  -/
  apply Set.eq_univ_of_forall fun f ↦ ?_
  simp only [Set.mem_preimage, toContinuousMapHom_apply, SetLike.mem_coe, RingHom.mem_ker,
    ContinuousMap.evalStarAlgHom_apply, ContinuousMap.coe_coe]
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    s : Set 𝕜
    inst✝¹ : Zero ↑s
    h0 : Eq (↑0) 0
    inst✝ : CompactSpace ↑s
    h0' : Membership.mem s 0
    f : ContinuousMapZero (↑s) 𝕜
    ⊢ Eq (f ⟨0, h0'⟩) 0
  -/
  rw [show ⟨0, h0'⟩ = (0 : s) by ext; exact h0.symm, _root_.map_zero f]
  /-
    🎉 no goals
  -/


/-- An induction principle for `C(s, 𝕜)₀`. -/
@[elab_as_elim]
lemma ContinuousMapZero.induction_on {s : Set 𝕜} [Zero s] (h0 : ((0 : s) : 𝕜) = 0)
    {p : C(s, 𝕜)₀ → Prop} (zero : p 0) (id : p (.id h0)) (star_id : p (star (.id h0)))
    (add : ∀ f g, p f → p g → p (f + g)) (mul : ∀ f g, p f → p g → p (f * g))
    (smul : ∀ (r : 𝕜) f, p f → p (r • f))
    (closure : (∀ f ∈ adjoin 𝕜 {(.id h0 : C(s, 𝕜)₀)}, p f) → ∀ f, p f) (f : C(s, 𝕜)₀) :
    p f := by
  /-
    𝕜 : Type u_1
    inst✝¹ : RCLike 𝕜
    s : Set 𝕜
    inst✝ : Zero ↑s
    h0 : Eq (↑0) 0
    p : ContinuousMapZero (↑s) 𝕜 → Prop
    zero : p 0
    id : p (ContinuousMapZero.id h0)
    star_id : p (Star.star (ContinuousMapZero.id h0))
    add : ∀ (f g : ContinuousMapZero (↑s) 𝕜), p f → p g → p (HAdd.hAdd f g)
    mul : ∀ (f g : ContinuousMapZero (↑s) 𝕜), p f → p g → p (HMul.hMul f g)
    smul : ∀ (r : 𝕜) (f : ContinuousMapZero (↑s) 𝕜), p f → p (HSMul.hSMul r f)
    closure : (∀ (f : ContinuousMapZero (↑s) 𝕜), Membership.mem (NonUnitalStarAlge …
    f : ContinuousMapZero (↑s) 𝕜
    ⊢ p f
  -/
  refine closure (fun f hf => ?_) f
  induction hf using NonUnitalAlgebra.adjoin_induction with
  | mem f hf =>
    simp only [Set.mem_union, Set.mem_singleton_iff, Set.mem_star] at hf
    rw [star_eq_iff_star_eq, eq_comm (b := f)] at hf
    obtain (rfl | rfl) := hf
    all_goals assumption
  | zero => exact zero
  | add _ _ _ _ hf hg => exact add _ _ hf hg
  | mul _ _ _ _ hf hg => exact mul _ _ hf hg
  | smul _ _ _ hf => exact smul _ _ hf


open Topology in
@[elab_as_elim]
theorem ContinuousMapZero.induction_on_of_compact {s : Set 𝕜} [Zero s] (h0 : ((0 : s) : 𝕜) = 0)
    [CompactSpace s] {p : C(s, 𝕜)₀ → Prop} (zero : p 0) (id : p (.id h0))
    (star_id : p (star (.id h0))) (add : ∀ f g, p f → p g → p (f + g))
    (mul : ∀ f g, p f → p g → p (f * g)) (smul : ∀ (r : 𝕜) f, p f → p (r • f))
    (frequently : ∀ f, (∃ᶠ g in 𝓝 f, p g) → p f) (f : C(s, 𝕜)₀) :
    p f := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    s : Set 𝕜
    inst✝¹ : Zero ↑s
    h0 : Eq (↑0) 0
    inst✝ : CompactSpace ↑s
    p : ContinuousMapZero (↑s) 𝕜 → Prop
    zero : p 0
    id : p (ContinuousMapZero.id h0)
    star_id : p (Star.star (ContinuousMapZero.id h0))
    add : ∀ (f g : ContinuousMapZero (↑s) 𝕜), p f → p g → p (HAdd.hAdd f g)
    mul : ∀ (f g : ContinuousMapZero (↑s) 𝕜), p f → p g → p (HMul.hMul f g)
    smul : ∀ (r : 𝕜) (f : ContinuousMapZero (↑s) 𝕜), p f → p (HSMul.hSMul r f)
    frequently : ∀ (f : ContinuousMapZero (↑s) 𝕜), Filter.Frequently (fun g => p g …
    f : ContinuousMapZero (↑s) 𝕜
    ⊢ p f
  -/
  refine f.induction_on h0 zero id star_id add mul smul fun h f ↦ frequently f ?_
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    s : Set 𝕜
    inst✝¹ : Zero ↑s
    h0 : Eq (↑0) 0
    inst✝ : CompactSpace ↑s
    p : ContinuousMapZero (↑s) 𝕜 → Prop
    zero : p 0
    id : p (ContinuousMapZero.id h0)
    star_id : p (Star.star (ContinuousMapZero.id h0))
    add : ∀ (f g : ContinuousMapZero (↑s) 𝕜), p f → p g → p (HAdd.hAdd f g)
    mul : ∀ (f g : ContinuousMapZero (↑s) 𝕜), p f → p g → p (HMul.hMul f g)
    smul : ∀ (r : 𝕜) (f : ContinuousMapZero (↑s) 𝕜), p f → p (HSMul.hSMul r f)
    frequently : ∀ (f : ContinuousMapZero (↑s) 𝕜), Filter.Frequently (fun g => p g …
    f✝ : ContinuousMapZero (↑s) 𝕜
    h : ∀ (f : ContinuousMapZero (↑s) 𝕜), Membership.mem (NonUnitalStarAlgebra.adj …
    f : ContinuousMapZero (↑s) 𝕜
    ⊢ Filter.Frequently (fun g => p g) (nhds f)
  -/
  have := (ContinuousMapZero.adjoin_id_dense h0).closure_eq ▸ Set.mem_univ (x := f)
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    s : Set 𝕜
    inst✝¹ : Zero ↑s
    h0 : Eq (↑0) 0
    inst✝ : CompactSpace ↑s
    p : ContinuousMapZero (↑s) 𝕜 → Prop
    zero : p 0
    id : p (ContinuousMapZero.id h0)
    star_id : p (Star.star (ContinuousMapZero.id h0))
    add : ∀ (f g : ContinuousMapZero (↑s) 𝕜), p f → p g → p (HAdd.hAdd f g)
    mul : ∀ (f g : ContinuousMapZero (↑s) 𝕜), p f → p g → p (HMul.hMul f g)
    smul : ∀ (r : 𝕜) (f : ContinuousMapZero (↑s) 𝕜), p f → p (HSMul.hSMul r f)
    frequently : ∀ (f : ContinuousMapZero (↑s) 𝕜), Filter.Frequently (fun g => p g …
    f✝ : ContinuousMapZero (↑s) 𝕜
    h : ∀ (f : ContinuousMapZero (↑s) 𝕜), Membership.mem (NonUnitalStarAlgebra.adj …
    f : ContinuousMapZero (↑s) 𝕜
    this : Membership.mem (closure ↑(NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.sing …
    ⊢ Filter.Frequently (fun g => p g) (nhds f)
  -/
  exact mem_closure_iff_frequently.mp this |>.mp <| .of_forall h
  /-
    🎉 no goals
  -/


lemma ContinuousMapZero.nonUnitalStarAlgHom_apply_mul_eq_zero {𝕜 A : Type*}
    [RCLike 𝕜] [NonUnitalRing A] [StarRing A] [TopologicalSpace A] [TopologicalSemiring A]
    [T2Space A] [Module 𝕜 A] [IsScalarTower 𝕜 A A] {s : Set 𝕜} [Zero s] [CompactSpace s]
    (h0 : (0 : s) = (0 : 𝕜)) (φ : C(s, 𝕜)₀ →⋆ₙₐ[𝕜] A) (a : A) (hmul_id : φ (.id h0) * a = 0)
    (hmul_star_id : φ (star (.id h0)) * a = 0) (hφ : Continuous φ) (f : C(s, 𝕜)₀) :
    φ f * a = 0 := by
  induction f using ContinuousMapZero.induction_on_of_compact h0 with
  | zero => simp [map_zero]
  | id => exact hmul_id
  | star_id => exact hmul_star_id
  | add _ _ h₁ h₂ => simp only [map_add, add_mul, h₁, h₂, zero_add]
  | mul _ _ _ h => simp only [map_mul, mul_assoc, h, mul_zero]
  | smul _ _ h => rw [map_smul, smul_mul_assoc, h, smul_zero]
  | frequently f h => exact h.mem_of_closed <| isClosed_eq (by fun_prop) continuous_zero


lemma ContinuousMapZero.mul_nonUnitalStarAlgHom_apply_eq_zero {𝕜 A : Type*}
    [RCLike 𝕜] [NonUnitalRing A] [StarRing A] [TopologicalSpace A] [TopologicalSemiring A]
    [T2Space A] [Module 𝕜 A] [SMulCommClass 𝕜 A A] {s : Set 𝕜} [Zero s] [CompactSpace s]
    (h0 : (0 : s) = (0 : 𝕜)) (φ : C(s, 𝕜)₀ →⋆ₙₐ[𝕜] A) (a : A) (hmul_id : a * φ (.id h0) = 0)
    (hmul_star_id : a * φ (star (.id h0)) = 0) (hφ : Continuous φ) (f : C(s, 𝕜)₀) :
    a * φ f = 0 := by
  induction f using ContinuousMapZero.induction_on_of_compact h0 with
  | zero => simp [map_zero]
  | id => exact hmul_id
  | star_id => exact hmul_star_id
  | add _ _ h₁ h₂ => simp only [map_add, mul_add, h₁, h₂, zero_add]
  | mul _ _ h _ => simp only [map_mul, ← mul_assoc, h, zero_mul]
  | smul _ _ h => rw [map_smul, mul_smul_comm, h, smul_zero]
  | frequently f h => exact h.mem_of_closed <| isClosed_eq (by fun_prop) continuous_zero


