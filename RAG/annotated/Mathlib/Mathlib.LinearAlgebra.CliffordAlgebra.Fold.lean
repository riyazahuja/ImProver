/-- Fold a bilinear map along the generators of a term of the clifford algebra, with the rule
given by `foldr Q f hf n (ι Q m * x) = f m (foldr Q f hf n x)`.

For example, `foldr f hf n (r • ι R u + ι R v * ι R w) = r • f u n + f v (f w n)`. -/
def foldr (f : M →ₗ[R] N →ₗ[R] N) (hf : ∀ m x, f m (f m x) = Q m • x) :
    N →ₗ[R] CliffordAlgebra Q →ₗ[R] N :=
  (CliffordAlgebra.lift Q ⟨f, fun v => LinearMap.ext <| hf v⟩).toLinearMap.flip


@[simp]
theorem foldr_ι (f : M →ₗ[R] N →ₗ[R] N) (hf) (n : N) (m : M) : foldr Q f hf n (ι Q m) = f m n :=
  LinearMap.congr_fun (lift_ι_apply _ _ _) n


@[simp]
theorem foldr_algebraMap (f : M →ₗ[R] N →ₗ[R] N) (hf) (n : N) (r : R) :
    foldr Q f hf n (algebraMap R _ r) = r • n :=
  LinearMap.congr_fun (AlgHom.commutes _ r) n


@[simp]
theorem foldr_one (f : M →ₗ[R] N →ₗ[R] N) (hf) (n : N) : foldr Q f hf n 1 = n :=
  LinearMap.congr_fun (map_one (lift Q _)) n


@[simp]
theorem foldr_mul (f : M →ₗ[R] N →ₗ[R] N) (hf) (n : N) (a b : CliffordAlgebra Q) :
    foldr Q f hf n (a * b) = foldr Q f hf (foldr Q f hf n b) a :=
  LinearMap.congr_fun (map_mul (lift Q _) _ _) n


/-- This lemma demonstrates the origin of the `foldr` name. -/
theorem foldr_prod_map_ι (l : List M) (f : M →ₗ[R] N →ₗ[R] N) (hf) (n : N) :
    foldr Q f hf n (l.map <| ι Q).prod = List.foldr (fun m n => f m n) n l := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    l : List M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N N)
    hf : ∀ (m : M) (x : N), Eq ((f m) ((f m) x)) (HSMul.hSMul (Q m) x)
    n : N
    ⊢ Eq (((CliffordAlgebra.foldr Q f hf) n) (List.map (⇑(CliffordAlgebra.ι Q)) l) …
  -/
  induction' l with hd tl ih
    /-
      case nil
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      Q : QuadraticForm R M
      f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N N)
      hf : ∀ (m : M) (x : N), Eq ((f m) ((f m) x)) (HSMul.hSMul (Q m) x)
      n : N
      ⊢ Eq (((CliffordAlgebra.foldr Q f hf) n) (List.map (⇑(CliffordAlgebra.ι Q)) Li …
    -/
  · rw [List.map_nil, List.prod_nil, List.foldr_nil, foldr_one]
    /-
      🎉 no goals
    -/
    /-
      case cons
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      Q : QuadraticForm R M
      f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N N)
      hf : ∀ (m : M) (x : N), Eq ((f m) ((f m) x)) (HSMul.hSMul (Q m) x)
      n : N
      hd : M
      tl : List M
      ih : Eq (((CliffordAlgebra.foldr Q f hf) n) (List.map (⇑(CliffordAlgebra.ι Q)) …
      ⊢ Eq (((CliffordAlgebra.foldr Q f hf) n) (List.map (⇑(CliffordAlgebra.ι Q)) (L …
    -/
  · rw [List.map_cons, List.prod_cons, List.foldr_cons, foldr_mul, foldr_ι, ih]
    /-
      🎉 no goals
    -/


/-- Fold a bilinear map along the generators of a term of the clifford algebra, with the rule
given by `foldl Q f hf n (ι Q m * x) = f m (foldl Q f hf n x)`.

For example, `foldl f hf n (r • ι R u + ι R v * ι R w) = r • f u n + f v (f w n)`. -/
def foldl (f : M →ₗ[R] N →ₗ[R] N) (hf : ∀ m x, f m (f m x) = Q m • x) :
    N →ₗ[R] CliffordAlgebra Q →ₗ[R] N :=
  LinearMap.compl₂ (foldr Q f hf) reverse


@[simp]
theorem foldl_reverse (f : M →ₗ[R] N →ₗ[R] N) (hf) (n : N) (x : CliffordAlgebra Q) :
    foldl Q f hf n (reverse x) = foldr Q f hf n x :=
  DFunLike.congr_arg (foldr Q f hf n) <| reverse_reverse _


@[simp]
theorem foldr_reverse (f : M →ₗ[R] N →ₗ[R] N) (hf) (n : N) (x : CliffordAlgebra Q) :
    foldr Q f hf n (reverse x) = foldl Q f hf n x :=
  rfl


@[simp]
theorem foldl_ι (f : M →ₗ[R] N →ₗ[R] N) (hf) (n : N) (m : M) : foldl Q f hf n (ι Q m) = f m n := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N N)
    hf : ∀ (m : M) (x : N), Eq ((f m) ((f m) x)) (HSMul.hSMul (Q m) x)
    n : N
    m : M
    ⊢ Eq (((CliffordAlgebra.foldl Q f hf) n) ((CliffordAlgebra.ι Q) m)) ((f m) n)
  -/
  rw [← foldr_reverse, reverse_ι, foldr_ι]
  /-
    🎉 no goals
  -/


@[simp]
theorem foldl_algebraMap (f : M →ₗ[R] N →ₗ[R] N) (hf) (n : N) (r : R) :
    foldl Q f hf n (algebraMap R _ r) = r • n := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N N)
    hf : ∀ (m : M) (x : N), Eq ((f m) ((f m) x)) (HSMul.hSMul (Q m) x)
    n : N
    r : R
    ⊢ Eq (((CliffordAlgebra.foldl Q f hf) n) ((algebraMap R (CliffordAlgebra Q)) r …
  -/
  rw [← foldr_reverse, reverse.commutes, foldr_algebraMap]
  /-
    🎉 no goals
  -/


@[simp]
theorem foldl_one (f : M →ₗ[R] N →ₗ[R] N) (hf) (n : N) : foldl Q f hf n 1 = n := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N N)
    hf : ∀ (m : M) (x : N), Eq ((f m) ((f m) x)) (HSMul.hSMul (Q m) x)
    n : N
    ⊢ Eq (((CliffordAlgebra.foldl Q f hf) n) 1) n
  -/
  rw [← foldr_reverse, reverse.map_one, foldr_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem foldl_mul (f : M →ₗ[R] N →ₗ[R] N) (hf) (n : N) (a b : CliffordAlgebra Q) :
    foldl Q f hf n (a * b) = foldl Q f hf (foldl Q f hf n a) b := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N N)
    hf : ∀ (m : M) (x : N), Eq ((f m) ((f m) x)) (HSMul.hSMul (Q m) x)
    n : N
    a b : CliffordAlgebra Q
    ⊢ Eq (((CliffordAlgebra.foldl Q f hf) n) (HMul.hMul a b)) (((CliffordAlgebra.f …
  -/
  rw [← foldr_reverse, ← foldr_reverse, ← foldr_reverse, reverse.map_mul, foldr_mul]
  /-
    🎉 no goals
  -/


/-- This lemma demonstrates the origin of the `foldl` name. -/
theorem foldl_prod_map_ι (l : List M) (f : M →ₗ[R] N →ₗ[R] N) (hf) (n : N) :
    foldl Q f hf n (l.map <| ι Q).prod = List.foldl (fun m n => f n m) n l := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    l : List M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N N)
    hf : ∀ (m : M) (x : N), Eq ((f m) ((f m) x)) (HSMul.hSMul (Q m) x)
    n : N
    ⊢ Eq (((CliffordAlgebra.foldl Q f hf) n) (List.map (⇑(CliffordAlgebra.ι Q)) l) …
  -/
  rw [← foldr_reverse, reverse_prod_map_ι, ← List.map_reverse, foldr_prod_map_ι, List.foldr_reverse]
  /-
    🎉 no goals
  -/


@[elab_as_elim]
theorem right_induction {P : CliffordAlgebra Q → Prop} (algebraMap : ∀ r : R, P (algebraMap _ _ r))
    (add : ∀ x y, P x → P y → P (x + y)) (mul_ι : ∀ m x, P x → P (x * ι Q m)) : ∀ x, P x := by
  /- It would be neat if we could prove this via `foldr` like how we prove
    `CliffordAlgebra.induction`, but going via the grading seems easier. -/
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    P : CliffordAlgebra Q → Prop
    algebraMap : ∀ (r : R), P ((_root_.algebraMap R (CliffordAlgebra Q)) r)
    add : ∀ (x y : CliffordAlgebra Q), P x → P y → P (HAdd.hAdd x y)
    mul_ι : ∀ (m : M) (x : CliffordAlgebra Q), P x → P (HMul.hMul x ((CliffordAlge …
    ⊢ ∀ (x : CliffordAlgebra Q), P x
  -/
  intro x
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    P : CliffordAlgebra Q → Prop
    algebraMap : ∀ (r : R), P ((_root_.algebraMap R (CliffordAlgebra Q)) r)
    add : ∀ (x y : CliffordAlgebra Q), P x → P y → P (HAdd.hAdd x y)
    mul_ι : ∀ (m : M) (x : CliffordAlgebra Q), P x → P (HMul.hMul x ((CliffordAlge …
    x : CliffordAlgebra Q
    ⊢ P x
  -/
  have : x ∈ ⊤ := Submodule.mem_top (R := R)
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    P : CliffordAlgebra Q → Prop
    algebraMap : ∀ (r : R), P ((_root_.algebraMap R (CliffordAlgebra Q)) r)
    add : ∀ (x y : CliffordAlgebra Q), P x → P y → P (HAdd.hAdd x y)
    mul_ι : ∀ (m : M) (x : CliffordAlgebra Q), P x → P (HMul.hMul x ((CliffordAlge …
    x : CliffordAlgebra Q
    this : Membership.mem Top.top x
    ⊢ P x
  -/
  rw [← iSup_ι_range_eq_top] at this
  induction this using Submodule.iSup_induction' with
  | mem i x hx =>
    induction hx using Submodule.pow_induction_on_right' with
    | algebraMap r => exact algebraMap r
    | add _x _y _i _ _ ihx ihy => exact add _ _ ihx ihy
    | mul_mem _i x _hx px m hm =>
      obtain ⟨m, rfl⟩ := hm
      exact mul_ι _ _ px
  | zero => simpa only [map_zero] using algebraMap 0
  | add _x _y _ _ ihx ihy =>
    exact add _ _ ihx ihy


@[elab_as_elim]
theorem left_induction {P : CliffordAlgebra Q → Prop} (algebraMap : ∀ r : R, P (algebraMap _ _ r))
    (add : ∀ x y, P x → P y → P (x + y)) (ι_mul : ∀ x m, P x → P (ι Q m * x)) : ∀ x, P x := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    P : CliffordAlgebra Q → Prop
    algebraMap : ∀ (r : R), P ((_root_.algebraMap R (CliffordAlgebra Q)) r)
    add : ∀ (x y : CliffordAlgebra Q), P x → P y → P (HAdd.hAdd x y)
    ι_mul : ∀ (x : CliffordAlgebra Q) (m : M), P x → P (HMul.hMul ((CliffordAlgebr …
    ⊢ ∀ (x : CliffordAlgebra Q), P x
  -/
  refine reverse_involutive.surjective.forall.2 ?_
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    P : CliffordAlgebra Q → Prop
    algebraMap : ∀ (r : R), P ((_root_.algebraMap R (CliffordAlgebra Q)) r)
    add : ∀ (x y : CliffordAlgebra Q), P x → P y → P (HAdd.hAdd x y)
    ι_mul : ∀ (x : CliffordAlgebra Q) (m : M), P x → P (HMul.hMul ((CliffordAlgebr …
    ⊢ ∀ (x : CliffordAlgebra Q), P (CliffordAlgebra.reverse x)
  -/
  intro x
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    P : CliffordAlgebra Q → Prop
    algebraMap : ∀ (r : R), P ((_root_.algebraMap R (CliffordAlgebra Q)) r)
    add : ∀ (x y : CliffordAlgebra Q), P x → P y → P (HAdd.hAdd x y)
    ι_mul : ∀ (x : CliffordAlgebra Q) (m : M), P x → P (HMul.hMul ((CliffordAlgebr …
    x : CliffordAlgebra Q
    ⊢ P (CliffordAlgebra.reverse x)
  -/
  induction' x using CliffordAlgebra.right_induction with r x y hx hy m x hx
    /-
      case algebraMap
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      P : CliffordAlgebra Q → Prop
      algebraMap : ∀ (r : R), P ((_root_.algebraMap R (CliffordAlgebra Q)) r)
      add : ∀ (x y : CliffordAlgebra Q), P x → P y → P (HAdd.hAdd x y)
      ι_mul : ∀ (x : CliffordAlgebra Q) (m : M), P x → P (HMul.hMul ((CliffordAlgebr …
      r : R
      ⊢ P (CliffordAlgebra.reverse ((_root_.algebraMap R (CliffordAlgebra Q)) r))
    -/
  · simpa only [reverse.commutes] using algebraMap r
    /-
      🎉 no goals
    -/
    /-
      case add
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      P : CliffordAlgebra Q → Prop
      algebraMap : ∀ (r : R), P ((_root_.algebraMap R (CliffordAlgebra Q)) r)
      add : ∀ (x y : CliffordAlgebra Q), P x → P y → P (HAdd.hAdd x y)
      ι_mul : ∀ (x : CliffordAlgebra Q) (m : M), P x → P (HMul.hMul ((CliffordAlgebr …
      x y : CliffordAlgebra Q
      hx : P (CliffordAlgebra.reverse x)
      hy : P (CliffordAlgebra.reverse y)
      ⊢ P (CliffordAlgebra.reverse (HAdd.hAdd x y))
    -/
  · simpa only [map_add] using add _ _ hx hy
    /-
      🎉 no goals
    -/
    /-
      case mul_ι
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      Q : QuadraticForm R M
      P : CliffordAlgebra Q → Prop
      algebraMap : ∀ (r : R), P ((_root_.algebraMap R (CliffordAlgebra Q)) r)
      add : ∀ (x y : CliffordAlgebra Q), P x → P y → P (HAdd.hAdd x y)
      ι_mul : ∀ (x : CliffordAlgebra Q) (m : M), P x → P (HMul.hMul ((CliffordAlgebr …
      m : M
      x : CliffordAlgebra Q
      hx : P (CliffordAlgebra.reverse x)
      ⊢ P (CliffordAlgebra.reverse (HMul.hMul x ((CliffordAlgebra.ι Q) m)))
    -/
  · simpa only [reverse.map_mul, reverse_ι] using ι_mul _ _ hx
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `CliffordAlgebra.foldr'` -/
def foldr'Aux (f : M →ₗ[R] CliffordAlgebra Q × N →ₗ[R] N) :
    M →ₗ[R] Module.End R (CliffordAlgebra Q × N) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) (Prod (CliffordAlgebr …
    ⊢ LinearMap (RingHom.id R) M (Module.End R (Prod (CliffordAlgebra Q) N))
  -/
  have v_mul := (Algebra.lmul R (CliffordAlgebra Q)).toLinearMap ∘ₗ ι Q
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) (Prod (CliffordAlgebr …
    v_mul : LinearMap (RingHom.id R) M (Module.End R (CliffordAlgebra Q))
    ⊢ LinearMap (RingHom.id R) M (Module.End R (Prod (CliffordAlgebra Q) N))
  -/
  have l := v_mul.compl₂ (LinearMap.fst _ _ N)
  exact
    { toFun := fun m => (l m).prod (f m)
      map_add' := fun v₂ v₂ =>
        LinearMap.ext fun x =>
          Prod.ext (LinearMap.congr_fun (l.map_add _ _) x) (LinearMap.congr_fun (f.map_add _ _) x)
      map_smul' := fun c v =>
        LinearMap.ext fun x =>
          Prod.ext (LinearMap.congr_fun (l.map_smul _ _) x)
            (LinearMap.congr_fun (f.map_smul _ _) x) }


theorem foldr'Aux_apply_apply (f : M →ₗ[R] CliffordAlgebra Q × N →ₗ[R] N) (m : M) (x_fx) :
    foldr'Aux Q f m x_fx = (ι Q m * x_fx.1, f m x_fx) :=
  rfl


theorem foldr'Aux_foldr'Aux (f : M →ₗ[R] CliffordAlgebra Q × N →ₗ[R] N)
    (hf : ∀ m x fx, f m (ι Q m * x, f m (x, fx)) = Q m • fx) (v : M) (x_fx) :
    foldr'Aux Q f v (foldr'Aux Q f v x_fx) = Q v • x_fx := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) (Prod (CliffordAlgebr …
    hf : ∀ (m : M) (x : CliffordAlgebra Q) (fx : N), Eq ((f m) { fst := HMul.hMul  …
    v : M
    x_fx : Prod (CliffordAlgebra Q) N
    ⊢ Eq (((CliffordAlgebra.foldr'Aux Q f) v) (((CliffordAlgebra.foldr'Aux Q f) v) …
  -/
  cases' x_fx with x fx
  /-
    case mk
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) (Prod (CliffordAlgebr …
    hf : ∀ (m : M) (x : CliffordAlgebra Q) (fx : N), Eq ((f m) { fst := HMul.hMul  …
    v : M
    x : CliffordAlgebra Q
    fx : N
    ⊢ Eq (((CliffordAlgebra.foldr'Aux Q f) v) (((CliffordAlgebra.foldr'Aux Q f) v) …
  -/
  simp only [foldr'Aux_apply_apply]
  /-
    case mk
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) (Prod (CliffordAlgebr …
    hf : ∀ (m : M) (x : CliffordAlgebra Q) (fx : N), Eq ((f m) { fst := HMul.hMul  …
    v : M
    x : CliffordAlgebra Q
    fx : N
    ⊢ Eq { fst := HMul.hMul ((CliffordAlgebra.ι Q) v) (HMul.hMul ((CliffordAlgebra …
  -/
  rw [← mul_assoc, ι_sq_scalar, ← Algebra.smul_def, hf, Prod.smul_mk]
  /-
    🎉 no goals
  -/


/-- Fold a bilinear map along the generators of a term of the clifford algebra, with the rule
given by `foldr' Q f hf n (ι Q m * x) = f m (x, foldr' Q f hf n x)`.
Note this is like `CliffordAlgebra.foldr`, but with an extra `x` argument.
Implement the recursion scheme `F[n0](m * x) = f(m, (x, F[n0](x)))`. -/
def foldr' (f : M →ₗ[R] CliffordAlgebra Q × N →ₗ[R] N)
    (hf : ∀ m x fx, f m (ι Q m * x, f m (x, fx)) = Q m • fx) (n : N) : CliffordAlgebra Q →ₗ[R] N :=
  LinearMap.snd _ _ _ ∘ₗ foldr Q (foldr'Aux Q f) (foldr'Aux_foldr'Aux Q _ hf) (1, n)


theorem foldr'_algebraMap (f : M →ₗ[R] CliffordAlgebra Q × N →ₗ[R] N)
    (hf : ∀ m x fx, f m (ι Q m * x, f m (x, fx)) = Q m • fx) (n r) :
    foldr' Q f hf n (algebraMap R _ r) = r • n :=
  congr_arg Prod.snd (foldr_algebraMap _ _ _ _ _)


theorem foldr'_ι (f : M →ₗ[R] CliffordAlgebra Q × N →ₗ[R] N)
    (hf : ∀ m x fx, f m (ι Q m * x, f m (x, fx)) = Q m • fx) (n m) :
    foldr' Q f hf n (ι Q m) = f m (1, n) :=
  congr_arg Prod.snd (foldr_ι _ _ _ _ _)


theorem foldr'_ι_mul (f : M →ₗ[R] CliffordAlgebra Q × N →ₗ[R] N)
    (hf : ∀ m x fx, f m (ι Q m * x, f m (x, fx)) = Q m • fx) (n m) (x) :
    foldr' Q f hf n (ι Q m * x) = f m (x, foldr' Q f hf n x) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) (Prod (CliffordAlgebr …
    hf : ∀ (m : M) (x : CliffordAlgebra Q) (fx : N), Eq ((f m) { fst := HMul.hMul  …
    n : N
    m : M
    x : CliffordAlgebra Q
    ⊢ Eq ((CliffordAlgebra.foldr' Q f hf n) (HMul.hMul ((CliffordAlgebra.ι Q) m) x …
  -/
  dsimp [foldr']
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) (Prod (CliffordAlgebr …
    hf : ∀ (m : M) (x : CliffordAlgebra Q) (fx : N), Eq ((f m) { fst := HMul.hMul  …
    n : N
    m : M
    x : CliffordAlgebra Q
    ⊢ Eq (((CliffordAlgebra.foldr Q (CliffordAlgebra.foldr'Aux Q f) ⋯) { fst := 1, …
  -/
  rw [foldr_mul, foldr_ι, foldr'Aux_apply_apply]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) (Prod (CliffordAlgebr …
    hf : ∀ (m : M) (x : CliffordAlgebra Q) (fx : N), Eq ((f m) { fst := HMul.hMul  …
    n : N
    m : M
    x : CliffordAlgebra Q
    ⊢ Eq { fst := HMul.hMul ((CliffordAlgebra.ι Q) m) (((CliffordAlgebra.foldr Q ( …
  -/
  refine congr_arg (f m) (Prod.mk.eta.symm.trans ?_)
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticForm R M
    f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) (Prod (CliffordAlgebr …
    hf : ∀ (m : M) (x : CliffordAlgebra Q) (fx : N), Eq ((f m) { fst := HMul.hMul  …
    n : N
    m : M
    x : CliffordAlgebra Q
    ⊢ Eq { fst := (((CliffordAlgebra.foldr Q (CliffordAlgebra.foldr'Aux Q f) ⋯) {  …
  -/
  congr 1
  induction x using CliffordAlgebra.left_induction with
  | algebraMap r => simp_rw [foldr_algebraMap, Prod.smul_mk, Algebra.algebraMap_eq_smul_one]
  | add x y hx hy => rw [map_add, Prod.fst_add, hx, hy]
  | ι_mul m x hx => rw [foldr_mul, foldr_ι, foldr'Aux_apply_apply, hx]


