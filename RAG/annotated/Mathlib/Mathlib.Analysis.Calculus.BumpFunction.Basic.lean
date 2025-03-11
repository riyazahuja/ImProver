/-- `f : ContDiffBump c`, where `c` is a point in a normed vector space, is a
bundled smooth function such that

- `f` is equal to `1` in `Metric.closedBall c f.rIn`;
- `support f = Metric.ball c f.rOut`;
- `0 ≤ f x ≤ 1` for all `x`.

The structure `ContDiffBump` contains the data required to construct the function:
real numbers `rIn`, `rOut`, and proofs of `0 < rIn < rOut`. The function itself is available through
`CoeFun` when the space is nice enough, i.e., satisfies the `HasContDiffBump` typeclass. -/
structure ContDiffBump (c : E) where
  /-- real numbers `0 < rIn < rOut` -/
  (rIn rOut : ℝ)
  rIn_pos : 0 < rIn
  rIn_lt_rOut : rIn < rOut


/-- The base function from which one will construct a family of bump functions. One could
add more properties if they are useful and satisfied in the examples of inner product spaces
and finite dimensional vector spaces, notably derivative norm control in terms of `R - 1`.

TODO: do we ever need `f x = 1 ↔ ‖x‖ ≤ 1`? -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not yet ported; was @[nolint has_nonempty_instance]
structure ContDiffBumpBase (E : Type*) [NormedAddCommGroup E] [NormedSpace ℝ E] where
  /-- The function underlying this family of bump functions -/
  toFun : ℝ → E → ℝ
  mem_Icc : ∀ (R : ℝ) (x : E), toFun R x ∈ Icc (0 : ℝ) 1
  symmetric : ∀ (R : ℝ) (x : E), toFun R (-x) = toFun R x
  smooth : ContDiffOn ℝ ∞ (uncurry toFun) (Ioi (1 : ℝ) ×ˢ (univ : Set E))
  eq_one : ∀ R : ℝ, 1 < R → ∀ x : E, ‖x‖ ≤ 1 → toFun R x = 1
  support : ∀ R : ℝ, 1 < R → Function.support (toFun R) = Metric.ball (0 : E) R


/-- A class registering that a real vector space admits bump functions. This will be instantiated
first for inner product spaces, and then for finite-dimensional normed spaces.
We use a specific class instead of `Nonempty (ContDiffBumpBase E)` for performance reasons. -/
class HasContDiffBump (E : Type*) [NormedAddCommGroup E] [NormedSpace ℝ E] : Prop where
  out : Nonempty (ContDiffBumpBase E)


/-- In a space with `C^∞` bump functions, register some function that will be used as a basis
to construct bump functions of arbitrary size around any point. -/
def someContDiffBumpBase (E : Type*) [NormedAddCommGroup E] [NormedSpace ℝ E]
    [hb : HasContDiffBump E] : ContDiffBumpBase E :=
  Nonempty.some hb.out


theorem rOut_pos {c : E} (f : ContDiffBump c) : 0 < f.rOut :=
  f.rIn_pos.trans f.rIn_lt_rOut


theorem one_lt_rOut_div_rIn {c : E} (f : ContDiffBump c) : 1 < f.rOut / f.rIn := by
  /-
    E : Type u_1
    c : E
    f : ContDiffBump c
    ⊢ LT.lt 1 (HDiv.hDiv f.rOut f.rIn)
  -/
  rw [one_lt_div f.rIn_pos]
  /-
    E : Type u_1
    c : E
    f : ContDiffBump c
    ⊢ LT.lt f.rIn f.rOut
  -/
  exact f.rIn_lt_rOut
  /-
    🎉 no goals
  -/


instance (c : E) : Inhabited (ContDiffBump c) :=
  ⟨⟨1, 2, zero_lt_one, one_lt_two⟩⟩


/-- The function defined by `f : ContDiffBump c`. Use automatic coercion to
function instead. -/
@[coe] def toFun {c : E} (f : ContDiffBump c) : E → ℝ :=
  (someContDiffBumpBase E).toFun (f.rOut / f.rIn) ∘ fun x ↦ (f.rIn⁻¹ • (x - c))


instance : CoeFun (ContDiffBump c) fun _ => E → ℝ :=
  ⟨toFun⟩


protected theorem apply (x : E) :
    f x = (someContDiffBumpBase E).toFun (f.rOut / f.rIn) (f.rIn⁻¹ • (x - c)) :=
  rfl


protected theorem sub (x : E) : f (c - x) = f (c + x) := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : HasContDiffBump E
    c : E
    f : ContDiffBump c
    x : E
    ⊢ Eq (↑f (HSub.hSub c x)) (↑f (HAdd.hAdd c x))
  -/
  simp [f.apply, ContDiffBumpBase.symmetric]
  /-
    🎉 no goals
  -/


protected theorem neg (f : ContDiffBump (0 : E)) (x : E) : f (-x) = f x := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : HasContDiffBump E
    f : ContDiffBump 0
    x : E
    ⊢ Eq (↑f (Neg.neg x)) (↑f x)
  -/
  simp_rw [← zero_sub, f.sub, zero_add]
  /-
    🎉 no goals
  -/


theorem one_of_mem_closedBall (hx : x ∈ closedBall c f.rIn) : f x = 1 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : HasContDiffBump E
    c : E
    f : ContDiffBump c
    x : E
    hx : Membership.mem (Metric.closedBall c f.rIn) x
    ⊢ Eq (↑f x) 1
  -/
  apply ContDiffBumpBase.eq_one _ _ f.one_lt_rOut_div_rIn
  simpa only [norm_smul, Real.norm_eq_abs, abs_inv, abs_of_nonneg f.rIn_pos.le, ← div_eq_inv_mul,
    div_le_one f.rIn_pos] using mem_closedBall_iff_norm.1 hx


theorem nonneg : 0 ≤ f x :=
  (ContDiffBumpBase.mem_Icc (someContDiffBumpBase E) _ _).1


/-- A version of `ContDiffBump.nonneg` with `x` explicit -/
theorem nonneg' (x : E) : 0 ≤ f x := f.nonneg


theorem le_one : f x ≤ 1 :=
  (ContDiffBumpBase.mem_Icc (someContDiffBumpBase E) _ _).2


theorem support_eq : Function.support f = Metric.ball c f.rOut := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : HasContDiffBump E
    c : E
    f : ContDiffBump c
    ⊢ Eq (Function.support ↑f) (Metric.ball c f.rOut)
  -/
  simp only [toFun, support_comp_eq_preimage, ContDiffBumpBase.support _ _ f.one_lt_rOut_div_rIn]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : HasContDiffBump E
    c : E
    f : ContDiffBump c
    ⊢ Eq (Set.preimage (fun x => HSMul.hSMul (Inv.inv f.rIn) (HSub.hSub x c)) (Met …
  -/
  ext x
  simp only [mem_ball_iff_norm, sub_zero, norm_smul, mem_preimage, Real.norm_eq_abs, abs_inv,
    abs_of_pos f.rIn_pos, ← div_eq_inv_mul, div_lt_div_iff_of_pos_right f.rIn_pos]


theorem tsupport_eq : tsupport f = closedBall c f.rOut := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : HasContDiffBump E
    c : E
    f : ContDiffBump c
    ⊢ Eq (tsupport ↑f) (Metric.closedBall c f.rOut)
  -/
  simp_rw [tsupport, f.support_eq, closure_ball _ f.rOut_pos.ne']
  /-
    🎉 no goals
  -/


theorem pos_of_mem_ball (hx : x ∈ ball c f.rOut) : 0 < f x :=
                           /-
                             E : Type u_1
                             inst✝² : NormedAddCommGroup E
                             inst✝¹ : NormedSpace Real E
                             inst✝ : HasContDiffBump E
                             c : E
                             f : ContDiffBump c
                             x : E
                             hx : Membership.mem (Metric.ball c f.rOut) x
                             ⊢ Ne (↑f x) 0
                           -/
  f.nonneg.lt_of_ne' <| by rwa [← support_eq, mem_support] at hx
                           /-
                             🎉 no goals
                           -/


theorem zero_of_le_dist (hx : f.rOut ≤ dist x c) : f x = 0 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : HasContDiffBump E
    c : E
    f : ContDiffBump c
    x : E
    hx : LE.le f.rOut (Dist.dist x c)
    ⊢ Eq (↑f x) 0
  -/
  rwa [← nmem_support, support_eq, mem_ball, not_lt]
  /-
    🎉 no goals
  -/


protected theorem hasCompactSupport [FiniteDimensional ℝ E] : HasCompactSupport f := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : HasContDiffBump E
    c : E
    f : ContDiffBump c
    inst✝ : FiniteDimensional Real E
    ⊢ HasCompactSupport ↑f
  -/
  simp_rw [HasCompactSupport, f.tsupport_eq, isCompact_closedBall]
  /-
    🎉 no goals
  -/


theorem eventuallyEq_one_of_mem_ball (h : x ∈ ball c f.rIn) : f =ᶠ[𝓝 x] 1 :=
  mem_of_superset (closedBall_mem_nhds_of_mem h) fun _ ↦ f.one_of_mem_closedBall


theorem eventuallyEq_one : f =ᶠ[𝓝 c] 1 :=
  f.eventuallyEq_one_of_mem_ball (mem_ball_self f.rIn_pos)


/-- `ContDiffBump` is `𝒞ⁿ` in all its arguments. -/
protected theorem _root_.ContDiffWithinAt.contDiffBump {c g : X → E} {s : Set X}
    {f : ∀ x, ContDiffBump (c x)} {x : X} (hc : ContDiffWithinAt ℝ n c s x)
    (hr : ContDiffWithinAt ℝ n (fun x => (f x).rIn) s x)
    (hR : ContDiffWithinAt ℝ n (fun x => (f x).rOut) s x)
    (hg : ContDiffWithinAt ℝ n g s x) :
    ContDiffWithinAt ℝ n (fun x => f x (g x)) s x := by
  change ContDiffWithinAt ℝ n (uncurry (someContDiffBumpBase E).toFun ∘ fun x : X =>
    ((f x).rOut / (f x).rIn, (f x).rIn⁻¹ • (g x - c x))) s x
  refine (((someContDiffBumpBase E).smooth.contDiffAt ?_).of_le
    (mod_cast le_top)).comp_contDiffWithinAt x ?_
    /-
      case refine_1
      E : Type u_1
      X : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup X
      inst✝¹ : NormedSpace Real X
      inst✝ : HasContDiffBump E
      n : ENat
      c g : X → E
      s : Set X
      f : (x : X) → ContDiffBump (c x)
      x : X
      hc : ContDiffWithinAt Real (↑n) c s x
      hr : ContDiffWithinAt Real (↑n) (fun x => (f x).rIn) s x
      hR : ContDiffWithinAt Real (↑n) (fun x => (f x).rOut) s x
      hg : ContDiffWithinAt Real (↑n) g s x
      ⊢ Membership.mem (nhds { fst := HDiv.hDiv (f x).rOut (f x).rIn, snd := HSMul.h …
    -/
  · exact prod_mem_nhds (Ioi_mem_nhds (f x).one_lt_rOut_div_rIn) univ_mem
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      X : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup X
      inst✝¹ : NormedSpace Real X
      inst✝ : HasContDiffBump E
      n : ENat
      c g : X → E
      s : Set X
      f : (x : X) → ContDiffBump (c x)
      x : X
      hc : ContDiffWithinAt Real (↑n) c s x
      hr : ContDiffWithinAt Real (↑n) (fun x => (f x).rIn) s x
      hR : ContDiffWithinAt Real (↑n) (fun x => (f x).rOut) s x
      hg : ContDiffWithinAt Real (↑n) g s x
      ⊢ ContDiffWithinAt Real (↑n) (fun x => { fst := HDiv.hDiv (f x).rOut (f x).rIn …
    -/
  · exact (hR.div hr (f x).rIn_pos.ne').prod ((hr.inv (f x).rIn_pos.ne').smul (hg.sub hc))
    /-
      🎉 no goals
    -/


/-- `ContDiffBump` is `𝒞ⁿ` in all its arguments. -/
protected nonrec theorem _root_.ContDiffAt.contDiffBump {c g : X → E} {f : ∀ x, ContDiffBump (c x)}
    {x : X} (hc : ContDiffAt ℝ n c x) (hr : ContDiffAt ℝ n (fun x => (f x).rIn) x)
    (hR : ContDiffAt ℝ n (fun x => (f x).rOut) x) (hg : ContDiffAt ℝ n g x) :
    ContDiffAt ℝ n (fun x => f x (g x)) x :=
  hc.contDiffBump hr hR hg


theorem _root_.ContDiff.contDiffBump {c g : X → E} {f : ∀ x, ContDiffBump (c x)}
    (hc : ContDiff ℝ n c) (hr : ContDiff ℝ n fun x => (f x).rIn)
    (hR : ContDiff ℝ n fun x => (f x).rOut) (hg : ContDiff ℝ n g) :
    ContDiff ℝ n fun x => f x (g x) := by
  /-
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup X
    inst✝¹ : NormedSpace Real X
    inst✝ : HasContDiffBump E
    n : ENat
    c g : X → E
    f : (x : X) → ContDiffBump (c x)
    hc : ContDiff Real (↑n) c
    hr : ContDiff Real ↑n fun x => (f x).rIn
    hR : ContDiff Real ↑n fun x => (f x).rOut
    hg : ContDiff Real (↑n) g
    ⊢ ContDiff Real ↑n fun x => ↑(f x) (g x)
  -/
  rw [contDiff_iff_contDiffAt] at *
  /-
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup X
    inst✝¹ : NormedSpace Real X
    inst✝ : HasContDiffBump E
    n : ENat
    c g : X → E
    f : (x : X) → ContDiffBump (c x)
    hc : ∀ (x : X), ContDiffAt Real (↑n) c x
    hr : ∀ (x : X), ContDiffAt Real (↑n) (fun x => (f x).rIn) x
    hR : ∀ (x : X), ContDiffAt Real (↑n) (fun x => (f x).rOut) x
    hg : ∀ (x : X), ContDiffAt Real (↑n) g x
    ⊢ ∀ (x : X), ContDiffAt Real (↑n) (fun x => ↑(f x) (g x)) x
  -/
  exact fun x => (hc x).contDiffBump (hr x) (hR x) (hg x)
  /-
    🎉 no goals
  -/


protected theorem contDiff : ContDiff ℝ n f :=
  contDiff_const.contDiffBump contDiff_const contDiff_const contDiff_id


protected theorem contDiffAt : ContDiffAt ℝ n f x :=
  f.contDiff.contDiffAt


protected theorem contDiffWithinAt {s : Set E} : ContDiffWithinAt ℝ n f s x :=
  f.contDiffAt.contDiffWithinAt


protected theorem continuous : Continuous f :=
  contDiff_zero.mp f.contDiff


