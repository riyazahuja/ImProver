/-- An unbounded operator is closed iff its graph is closed. -/
def IsClosed (f : E →ₗ.[R] F) : Prop :=
  _root_.IsClosed (f.graph : Set (E × F))


/-- An unbounded operator is closable iff the closure of its graph is a graph. -/
def IsClosable (f : E →ₗ.[R] F) : Prop :=
  ∃ f' : LinearPMap R E F, f.graph.topologicalClosure = f'.graph


/-- A closed operator is trivially closable. -/
theorem IsClosed.isClosable {f : E →ₗ.[R] F} (hf : f.IsClosed) : f.IsClosable :=
  ⟨f, hf.submodule_topologicalClosure_eq⟩


/-- If `g` has a closable extension `f`, then `g` itself is closable. -/
theorem IsClosable.leIsClosable {f g : E →ₗ.[R] F} (hf : f.IsClosable) (hfg : g ≤ f) :
    g.IsClosable := by
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f g : LinearPMap R E F
    hf : f.IsClosable
    hfg : LE.le g f
    ⊢ g.IsClosable
  -/
  cases' hf with f' hf
  have : g.graph.topologicalClosure ≤ f'.graph := by
    rw [← hf]
    exact Submodule.topologicalClosure_mono (le_graph_of_le hfg)
  /-
    case intro
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f g : LinearPMap R E F
    hfg : LE.le g f
    f' : LinearPMap R E F
    hf : Eq f.graph.topologicalClosure f'.graph
    this : LE.le g.graph.topologicalClosure f'.graph
    ⊢ g.IsClosable
  -/
  use g.graph.topologicalClosure.toLinearPMap
  /-
    case h
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f g : LinearPMap R E F
    hfg : LE.le g f
    f' : LinearPMap R E F
    hf : Eq f.graph.topologicalClosure f'.graph
    this : LE.le g.graph.topologicalClosure f'.graph
    ⊢ Eq g.graph.topologicalClosure g.graph.topologicalClosure.toLinearPMap.graph
  -/
  rw [Submodule.toLinearPMap_graph_eq]
  /-
    case h.hg
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f g : LinearPMap R E F
    hfg : LE.le g f
    f' : LinearPMap R E F
    hf : Eq f.graph.topologicalClosure f'.graph
    this : LE.le g.graph.topologicalClosure f'.graph
    ⊢ ∀ (x : Prod E F), Membership.mem g.graph.topologicalClosure x → Eq x.1 0 → E …
  -/
  exact fun _ hx hx' => f'.graph_fst_eq_zero_snd (this hx) hx'
  /-
    🎉 no goals
  -/


/-- The closure is unique. -/
theorem IsClosable.existsUnique {f : E →ₗ.[R] F} (hf : f.IsClosable) :
    ∃! f' : E →ₗ.[R] F, f.graph.topologicalClosure = f'.graph := by
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : f.IsClosable
    ⊢ ExistsUnique fun f' => Eq f.graph.topologicalClosure f'.graph
  -/
  refine existsUnique_of_exists_of_unique hf fun _ _ hy₁ hy₂ => eq_of_eq_graph ?_
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : f.IsClosable
    x✝¹ x✝ : LinearPMap R E F
    hy₁ : Eq f.graph.topologicalClosure x✝¹.graph
    hy₂ : Eq f.graph.topologicalClosure x✝.graph
    ⊢ Eq x✝¹.graph x✝.graph
  -/
  rw [← hy₁, ← hy₂]
  /-
    🎉 no goals
  -/


open Classical in
/-- If `f` is closable, then `f.closure` is the closure. Otherwise it is defined
as `f.closure = f`. -/
noncomputable def closure (f : E →ₗ.[R] F) : E →ₗ.[R] F :=
  if hf : f.IsClosable then hf.choose else f


theorem closure_def {f : E →ₗ.[R] F} (hf : f.IsClosable) : f.closure = hf.choose := by
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : f.IsClosable
    ⊢ Eq f.closure (Exists.choose hf)
  -/
  simp [closure, hf]
  /-
    🎉 no goals
  -/


                                                                                 /-
                                                                                   R : Type u_1
                                                                                   E : Type u_2
                                                                                   F : Type u_3
                                                                                   inst✝¹¹ : CommRing R
                                                                                   inst✝¹⁰ : AddCommGroup E
                                                                                   inst✝⁹ : AddCommGroup F
                                                                                   inst✝⁸ : Module R E
                                                                                   inst✝⁷ : Module R F
                                                                                   inst✝⁶ : TopologicalSpace E
                                                                                   inst✝⁵ : TopologicalSpace F
                                                                                   inst✝⁴ : ContinuousAdd E
                                                                                   inst✝³ : ContinuousAdd F
                                                                                   inst✝² : TopologicalSpace R
                                                                                   inst✝¹ : ContinuousSMul R E
                                                                                   inst✝ : ContinuousSMul R F
                                                                                   f : LinearPMap R E F
                                                                                   hf : Not f.IsClosable
                                                                                   ⊢ Eq f.closure f
                                                                                 -/
theorem closure_def' {f : E →ₗ.[R] F} (hf : ¬f.IsClosable) : f.closure = f := by simp [closure, hf]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- The closure (as a submodule) of the graph is equal to the graph of the closure
  (as a `LinearPMap`). -/
theorem IsClosable.graph_closure_eq_closure_graph {f : E →ₗ.[R] F} (hf : f.IsClosable) :
    f.graph.topologicalClosure = f.closure.graph := by
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : f.IsClosable
    ⊢ Eq f.graph.topologicalClosure f.closure.graph
  -/
  rw [closure_def hf]
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : f.IsClosable
    ⊢ Eq f.graph.topologicalClosure (Exists.choose hf).graph
  -/
  exact hf.choose_spec
  /-
    🎉 no goals
  -/


/-- A `LinearPMap` is contained in its closure. -/
theorem le_closure (f : E →ₗ.[R] F) : f ≤ f.closure := by
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    ⊢ LE.le f f.closure
  -/
  by_cases hf : f.IsClosable
    /-
      case pos
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      hf : f.IsClosable
      ⊢ LE.le f f.closure
    -/
  · refine le_of_le_graph ?_
    /-
      case pos
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      hf : f.IsClosable
      ⊢ LE.le f.graph f.closure.graph
    -/
    rw [← hf.graph_closure_eq_closure_graph]
    /-
      case pos
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      hf : f.IsClosable
      ⊢ LE.le f.graph f.graph.topologicalClosure
    -/
    exact (graph f).le_topologicalClosure
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : Not f.IsClosable
    ⊢ LE.le f f.closure
  -/
  rw [closure_def' hf]
  /-
    🎉 no goals
  -/


theorem IsClosable.closure_mono {f g : E →ₗ.[R] F} (hg : g.IsClosable) (h : f ≤ g) :
    f.closure ≤ g.closure := by
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f g : LinearPMap R E F
    hg : g.IsClosable
    h : LE.le f g
    ⊢ LE.le f.closure g.closure
  -/
  refine le_of_le_graph ?_
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f g : LinearPMap R E F
    hg : g.IsClosable
    h : LE.le f g
    ⊢ LE.le f.closure.graph g.closure.graph
  -/
  rw [← (hg.leIsClosable h).graph_closure_eq_closure_graph]
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f g : LinearPMap R E F
    hg : g.IsClosable
    h : LE.le f g
    ⊢ LE.le f.graph.topologicalClosure g.closure.graph
  -/
  rw [← hg.graph_closure_eq_closure_graph]
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f g : LinearPMap R E F
    hg : g.IsClosable
    h : LE.le f g
    ⊢ LE.le f.graph.topologicalClosure g.graph.topologicalClosure
  -/
  exact Submodule.topologicalClosure_mono (le_graph_of_le h)
  /-
    🎉 no goals
  -/


/-- If `f` is closable, then the closure is closed. -/
theorem IsClosable.closure_isClosed {f : E →ₗ.[R] F} (hf : f.IsClosable) : f.closure.IsClosed := by
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : f.IsClosable
    ⊢ f.closure.IsClosed
  -/
  rw [IsClosed, ← hf.graph_closure_eq_closure_graph]
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : f.IsClosable
    ⊢ _root_.IsClosed ↑f.graph.topologicalClosure
  -/
  exact f.graph.isClosed_topologicalClosure
  /-
    🎉 no goals
  -/


/-- If `f` is closable, then the closure is closable. -/
theorem IsClosable.closureIsClosable {f : E →ₗ.[R] F} (hf : f.IsClosable) : f.closure.IsClosable :=
  hf.closure_isClosed.isClosable


theorem isClosable_iff_exists_closed_extension {f : E →ₗ.[R] F} :
    f.IsClosable ↔ ∃ g : E →ₗ.[R] F, g.IsClosed ∧ f ≤ g :=
  ⟨fun h => ⟨f.closure, h.closure_isClosed, f.le_closure⟩, fun ⟨_, hg, h⟩ =>
    hg.isClosable.leIsClosable h⟩


/-- A submodule `S` is a core of `f` if the closure of the restriction of `f` to `S` is `f`. -/
structure HasCore (f : E →ₗ.[R] F) (S : Submodule R E) : Prop where
  le_domain : S ≤ f.domain
  closure_eq : (f.domRestrict S).closure = f


theorem hasCore_def {f : E →ₗ.[R] F} {S : Submodule R E} (h : f.HasCore S) :
    (f.domRestrict S).closure = f :=
  h.2


/-- For every unbounded operator `f` the submodule `f.domain` is a core of its closure.

Note that we don't require that `f` is closable, due to the definition of the closure. -/
theorem closureHasCore (f : E →ₗ.[R] F) : f.closure.HasCore f.domain := by
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    ⊢ f.closure.HasCore f.domain
  -/
  refine ⟨f.le_closure.1, ?_⟩
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    ⊢ Eq (f.closure.domRestrict f.domain).closure f.closure
  -/
  congr
  /-
    case e_f
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    ⊢ Eq (f.closure.domRestrict f.domain) f
  -/
  ext x y hxy
    /-
      case e_f.h.h
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      x : E
      ⊢ Iff (Membership.mem (f.closure.domRestrict f.domain).domain x) (Membership.m …
    -/
  · simp only [domRestrict_domain, Submodule.mem_inf, and_iff_left_iff_imp]
    /-
      case e_f.h.h
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      x : E
      ⊢ Membership.mem f.domain x → Membership.mem f.closure.domain x
    -/
    intro hx
    /-
      case e_f.h.h
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      x : E
      hx : Membership.mem f.domain x
      ⊢ Membership.mem f.closure.domain x
    -/
    exact f.le_closure.1 hx
    /-
      🎉 no goals
    -/
  /-
    case e_f.h'
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    x : Subtype fun x => Membership.mem (f.closure.domRestrict f.domain).domain x
    y : Subtype fun x => Membership.mem f.domain x
    hxy : Eq ↑x ↑y
    ⊢ Eq (↑(f.closure.domRestrict f.domain) x) (↑f y)
  -/
  let z : f.closure.domain := ⟨y.1, f.le_closure.1 y.2⟩
  /-
    case e_f.h'
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    x : Subtype fun x => Membership.mem (f.closure.domRestrict f.domain).domain x
    y : Subtype fun x => Membership.mem f.domain x
    hxy : Eq ↑x ↑y
    z : Subtype fun x => Membership.mem f.closure.domain x := ⟨↑y, ⋯⟩
    ⊢ Eq (↑(f.closure.domRestrict f.domain) x) (↑f y)
  -/
  have hyz : (y : E) = z := by simp [z]
  /-
    case e_f.h'
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    x : Subtype fun x => Membership.mem (f.closure.domRestrict f.domain).domain x
    y : Subtype fun x => Membership.mem f.domain x
    hxy : Eq ↑x ↑y
    z : Subtype fun x => Membership.mem f.closure.domain x := ⟨↑y, ⋯⟩
    hyz : Eq ↑y ↑z
    ⊢ Eq (↑(f.closure.domRestrict f.domain) x) (↑f y)
  -/
  rw [f.le_closure.2 hyz]
  /-
    case e_f.h'
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    x : Subtype fun x => Membership.mem (f.closure.domRestrict f.domain).domain x
    y : Subtype fun x => Membership.mem f.domain x
    hxy : Eq ↑x ↑y
    z : Subtype fun x => Membership.mem f.closure.domain x := ⟨↑y, ⋯⟩
    hyz : Eq ↑y ↑z
    ⊢ Eq (↑(f.closure.domRestrict f.domain) x) (↑f.closure z)
  -/
  exact domRestrict_apply (hxy.trans hyz)
  /-
    🎉 no goals
  -/


/-- If `f` is invertible and closable as well as its closure being invertible, then
the graph of the inverse of the closure is given by the closure of the graph of the inverse. -/
theorem closure_inverse_graph (hf : LinearMap.ker f.toFun = ⊥) (hf' : f.IsClosable)
    (hcf : LinearMap.ker f.closure.toFun = ⊥) :
    f.closure.inverse.graph = f.inverse.graph.topologicalClosure := by
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    hf' : f.IsClosable
    hcf : Eq (LinearMap.ker f.closure.toFun) Bot.bot
    ⊢ Eq f.closure.inverse.graph f.inverse.graph.topologicalClosure
  -/
  rw [inverse_graph hf, inverse_graph hcf, ← hf'.graph_closure_eq_closure_graph]
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    hf' : f.IsClosable
    hcf : Eq (LinearMap.ker f.closure.toFun) Bot.bot
    ⊢ Eq (Submodule.map (LinearEquiv.prodComm R E F) f.graph.topologicalClosure) ( …
  -/
  apply SetLike.ext'
  /-
    case h
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    hf' : f.IsClosable
    hcf : Eq (LinearMap.ker f.closure.toFun) Bot.bot
    ⊢ Eq ↑(Submodule.map (LinearEquiv.prodComm R E F) f.graph.topologicalClosure)  …
  -/
  simp only [Submodule.topologicalClosure_coe, Submodule.map_coe, LinearEquiv.prodComm_apply]
  /-
    case h
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    hf' : f.IsClosable
    hcf : Eq (LinearMap.ker f.closure.toFun) Bot.bot
    ⊢ Eq (Set.image (fun a => a.swap) (_root_.closure ↑f.graph)) (_root_.closure ( …
  -/
  apply (image_closure_subset_closure_image continuous_swap).antisymm
  /-
    case h
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    hf' : f.IsClosable
    hcf : Eq (LinearMap.ker f.closure.toFun) Bot.bot
    ⊢ HasSubset.Subset (_root_.closure (Set.image Prod.swap ↑f.graph)) (Set.image  …
  -/
  have h1 := Set.image_equiv_eq_preimage_symm f.graph (LinearEquiv.prodComm R E F).toEquiv
  have h2 := Set.image_equiv_eq_preimage_symm (_root_.closure f.graph)
    (LinearEquiv.prodComm R E F).toEquiv
  simp only [LinearEquiv.coe_toEquiv, LinearEquiv.prodComm_apply,
    LinearEquiv.coe_toEquiv_symm] at h1 h2
  /-
    case h
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    hf' : f.IsClosable
    hcf : Eq (LinearMap.ker f.closure.toFun) Bot.bot
    h1 : Eq (Set.image (fun a => a.swap) ↑f.graph) (Set.preimage ⇑↑(LinearEquiv.pr …
    h2 : Eq (Set.image (fun a => a.swap) (_root_.closure ↑f.graph)) (Set.preimage  …
    ⊢ HasSubset.Subset (_root_.closure (Set.image Prod.swap ↑f.graph)) (Set.image  …
  -/
  rw [h1, h2]
  /-
    case h
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    hf' : f.IsClosable
    hcf : Eq (LinearMap.ker f.closure.toFun) Bot.bot
    h1 : Eq (Set.image (fun a => a.swap) ↑f.graph) (Set.preimage ⇑↑(LinearEquiv.pr …
    h2 : Eq (Set.image (fun a => a.swap) (_root_.closure ↑f.graph)) (Set.preimage  …
    ⊢ HasSubset.Subset (_root_.closure (Set.preimage ⇑↑(LinearEquiv.prodComm R E F …
  -/
  apply continuous_swap.closure_preimage_subset
  /-
    🎉 no goals
  -/


/-- Assuming that `f` is invertible and closable, then the closure is invertible if and only
if the inverse of `f` is closable. -/
theorem inverse_isClosable_iff (hf : LinearMap.ker f.toFun = ⊥) (hf' : f.IsClosable) :
    f.inverse.IsClosable ↔ LinearMap.ker f.closure.toFun = ⊥ := by
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    hf' : f.IsClosable
    ⊢ Iff f.inverse.IsClosable (Eq (LinearMap.ker f.closure.toFun) Bot.bot)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      hf : Eq (LinearMap.ker f.toFun) Bot.bot
      hf' : f.IsClosable
      ⊢ f.inverse.IsClosable → Eq (LinearMap.ker f.closure.toFun) Bot.bot
    -/
  · intro ⟨f', h⟩
    /-
      case mp
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      hf : Eq (LinearMap.ker f.toFun) Bot.bot
      hf' : f.IsClosable
      f' : LinearPMap R F E
      h : Eq f.inverse.graph.topologicalClosure f'.graph
      ⊢ Eq (LinearMap.ker f.closure.toFun) Bot.bot
    -/
    rw [LinearMap.ker_eq_bot']
    /-
      case mp
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      hf : Eq (LinearMap.ker f.toFun) Bot.bot
      hf' : f.IsClosable
      f' : LinearPMap R F E
      h : Eq f.inverse.graph.topologicalClosure f'.graph
      ⊢ ∀ (m : Subtype fun x => Membership.mem f.closure.domain x), Eq (f.closure.to …
    -/
    intro ⟨x, hx⟩ hx'
    /-
      case mp
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      hf : Eq (LinearMap.ker f.toFun) Bot.bot
      hf' : f.IsClosable
      f' : LinearPMap R F E
      h : Eq f.inverse.graph.topologicalClosure f'.graph
      x : E
      hx : Membership.mem f.closure.domain x
      hx' : Eq (f.closure.toFun ⟨x, hx⟩) 0
      ⊢ Eq ⟨x, hx⟩ 0
    -/
    simp only [Submodule.mk_eq_zero]
    /-
      case mp
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      hf : Eq (LinearMap.ker f.toFun) Bot.bot
      hf' : f.IsClosable
      f' : LinearPMap R F E
      h : Eq f.inverse.graph.topologicalClosure f'.graph
      x : E
      hx : Membership.mem f.closure.domain x
      hx' : Eq (f.closure.toFun ⟨x, hx⟩) 0
      ⊢ Eq x 0
    -/
    rw [toFun_eq_coe, eq_comm, image_iff] at hx'
    have : (0, x) ∈ graph f' := by
      rw [← h, inverse_graph hf]
      rw [← hf'.graph_closure_eq_closure_graph, ← SetLike.mem_coe,
        Submodule.topologicalClosure_coe] at hx'
      apply image_closure_subset_closure_image continuous_swap
      simp only [Set.mem_image, Prod.exists, Prod.swap_prod_mk, Prod.mk.injEq]
      exact ⟨x, 0, hx', rfl, rfl⟩
    /-
      case mp
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      hf : Eq (LinearMap.ker f.toFun) Bot.bot
      hf' : f.IsClosable
      f' : LinearPMap R F E
      h : Eq f.inverse.graph.topologicalClosure f'.graph
      x : E
      hx : Membership.mem f.closure.domain x
      hx' : Membership.mem f.closure.graph { fst := x, snd := 0 }
      this : Membership.mem f'.graph { fst := 0, snd := x }
      ⊢ Eq x 0
    -/
    exact graph_fst_eq_zero_snd f' this rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      hf : Eq (LinearMap.ker f.toFun) Bot.bot
      hf' : f.IsClosable
      ⊢ Eq (LinearMap.ker f.closure.toFun) Bot.bot → f.inverse.IsClosable
    -/
  · intro h
    /-
      case mpr
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      hf : Eq (LinearMap.ker f.toFun) Bot.bot
      hf' : f.IsClosable
      h : Eq (LinearMap.ker f.closure.toFun) Bot.bot
      ⊢ f.inverse.IsClosable
    -/
    use f.closure.inverse
    /-
      case h
      R : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : AddCommGroup E
      inst✝⁹ : AddCommGroup F
      inst✝⁸ : Module R E
      inst✝⁷ : Module R F
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : ContinuousAdd E
      inst✝³ : ContinuousAdd F
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousSMul R E
      inst✝ : ContinuousSMul R F
      f : LinearPMap R E F
      hf : Eq (LinearMap.ker f.toFun) Bot.bot
      hf' : f.IsClosable
      h : Eq (LinearMap.ker f.closure.toFun) Bot.bot
      ⊢ Eq f.inverse.graph.topologicalClosure f.closure.inverse.graph
    -/
    exact (closure_inverse_graph hf hf' h).symm
    /-
      🎉 no goals
    -/


/-- If `f` is invertible and closable, then taking the closure and the inverse commute. -/
theorem inverse_closure (hf : LinearMap.ker f.toFun = ⊥) (hf' : f.IsClosable)
    (hcf : LinearMap.ker f.closure.toFun = ⊥) :
    f.inverse.closure = f.closure.inverse := by
  /-
    R : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : AddCommGroup F
    inst✝⁸ : Module R E
    inst✝⁷ : Module R F
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : ContinuousAdd E
    inst✝³ : ContinuousAdd F
    inst✝² : TopologicalSpace R
    inst✝¹ : ContinuousSMul R E
    inst✝ : ContinuousSMul R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    hf' : f.IsClosable
    hcf : Eq (LinearMap.ker f.closure.toFun) Bot.bot
    ⊢ Eq f.inverse.closure f.closure.inverse
  -/
  apply eq_of_eq_graph
  rw [closure_inverse_graph hf hf' hcf,
    ((inverse_isClosable_iff hf hf').mpr hcf).graph_closure_eq_closure_graph]


