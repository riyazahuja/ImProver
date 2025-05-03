/-- Homogeneous tensor powers $M^{\otimes n}$. `⨂[R]^n M` is a shorthand for
`⨂[R] (i : Fin n), M`. -/
abbrev TensorPower (R : Type*) (n : ℕ) (M : Type*) [CommSemiring R] [AddCommMonoid M]
    [Module R M] : Type _ :=
  ⨂[R] _ : Fin n, M


@[inherit_doc] scoped[TensorProduct] notation:max "⨂[" R "]^" n:arg => TensorPower R n


/-- Two dependent pairs of tensor products are equal if their index is equal and the contents
are equal after a canonical reindexing. -/
@[ext (iff := false)]
theorem gradedMonoid_eq_of_reindex_cast {ιι : Type*} {ι : ιι → Type*} :
    ∀ {a b : GradedMonoid fun ii => ⨂[R] _ : ι ii, M} (h : a.fst = b.fst),
      reindex R (fun _ ↦ M) (Equiv.cast <| congr_arg ι h) a.snd = b.snd → a = b
  | ⟨ai, a⟩, ⟨bi, b⟩ => fun (hi : ai = bi) (h : reindex R (fun _ ↦ M) _ a = b) => by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ιι : Type u_3
      ι : ιι → Type u_4
      ai : ιι
      a : PiTensorProduct R fun x => M
      bi : ιι
      b : PiTensorProduct R fun x => M
      hi : Eq ai bi
      h : Eq ((PiTensorProduct.reindex R (fun x => M) (Equiv.cast ⋯)) a) b
      ⊢ Eq ⟨ai, a⟩ ⟨bi, b⟩
    -/
    subst hi
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ιι : Type u_3
      ι : ιι → Type u_4
      ai : ιι
      a b : PiTensorProduct R fun x => M
      h : Eq ((PiTensorProduct.reindex R (fun x => M) (Equiv.cast ⋯)) a) b
      ⊢ Eq ⟨ai, a⟩ ⟨ai, b⟩
    -/
    simp_all
    /-
      🎉 no goals
    -/


/-- As a graded monoid, `⨂[R]^i M` has a `1 : ⨂[R]^0 M`. -/
instance gOne : GradedMonoid.GOne fun i => ⨂[R]^i M where one := tprod R <| @Fin.elim0 M


local notation "ₜ1" => @GradedMonoid.GOne.one ℕ (fun i => ⨂[R]^i M) _ _


theorem gOne_def : ₜ1 = tprod R (@Fin.elim0 M) :=
  rfl


/-- A variant of `PiTensorProduct.tmulEquiv` with the result indexed by `Fin (n + m)`. -/
def mulEquiv {n m : ℕ} : ⨂[R]^n M ⊗[R] (⨂[R]^m) M ≃ₗ[R] (⨂[R]^(n + m)) M :=
  (tmulEquiv R M).trans (reindex R (fun _ ↦ M) finSumFinEquiv)


/-- As a graded monoid, `⨂[R]^i M` has a `(*) : ⨂[R]^i M → ⨂[R]^j M → ⨂[R]^(i + j) M`. -/
instance gMul : GradedMonoid.GMul fun i => ⨂[R]^i M where
  mul {i j} a b :=
    (TensorProduct.mk R _ _).compr₂ (↑(mulEquiv : _ ≃ₗ[R] (⨂[R]^(i + j)) M)) a b


local infixl:70 " ₜ* " => @GradedMonoid.GMul.mul ℕ (fun i => ⨂[R]^i M) _ _ _ _


theorem gMul_def {i j} (a : ⨂[R]^i M) (b : (⨂[R]^j) M) :
    a ₜ* b = @mulEquiv R M _ _ _ i j (a ⊗ₜ b) :=
  rfl


theorem gMul_eq_coe_linearMap {i j} (a : ⨂[R]^i M) (b : (⨂[R]^j) M) :
    a ₜ* b = ((TensorProduct.mk R _ _).compr₂ ↑(mulEquiv : _ ≃ₗ[R] (⨂[R]^(i + j)) M) :
      ⨂[R]^i M →ₗ[R] (⨂[R]^j) M →ₗ[R] (⨂[R]^(i + j)) M) a b :=
  rfl


/-- Cast between "equal" tensor powers. -/
def cast {i j} (h : i = j) : ⨂[R]^i M ≃ₗ[R] (⨂[R]^j) M := reindex R (fun _ ↦ M) (finCongr h)


theorem cast_tprod {i j} (h : i = j) (a : Fin i → M) :
    cast R M h (tprod R a) = tprod R (a ∘ Fin.cast h.symm) :=
  reindex_tprod _ _


@[simp]
theorem cast_refl {i} (h : i = i) : cast R M h = LinearEquiv.refl _ _ :=
  (congr_arg (reindex R fun _ ↦ M) <| finCongr_refl h).trans reindex_refl


@[simp]
theorem cast_symm {i j} (h : i = j) : (cast R M h).symm = cast R M h.symm :=
  reindex_symm _


@[simp]
theorem cast_trans {i j k} (h : i = j) (h' : j = k) :
    (cast R M h).trans (cast R M h') = cast R M (h.trans h') :=
  reindex_trans _ _


@[simp]
theorem cast_cast {i j k} (h : i = j) (h' : j = k) (a : ⨂[R]^i M) :
    cast R M h' (cast R M h a) = cast R M (h.trans h') a :=
  reindex_reindex _ _ _


@[ext (iff := false)]
theorem gradedMonoid_eq_of_cast {a b : GradedMonoid fun n => ⨂[R] _ : Fin n, M} (h : a.fst = b.fst)
    (h2 : cast R M h a.snd = b.snd) : a = b := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a b : GradedMonoid fun n => PiTensorProduct R fun x => M
    h : Eq a.fst b.fst
    h2 : Eq ((TensorPower.cast R M h) a.snd) b.snd
    ⊢ Eq a b
  -/
  refine gradedMonoid_eq_of_reindex_cast h ?_
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a b : GradedMonoid fun n => PiTensorProduct R fun x => M
    h : Eq a.fst b.fst
    h2 : Eq ((TensorPower.cast R M h) a.snd) b.snd
    ⊢ Eq ((PiTensorProduct.reindex R (fun x => M) (Equiv.cast ⋯)) a.snd) b.snd
  -/
  rw [cast] at h2
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a b : GradedMonoid fun n => PiTensorProduct R fun x => M
    h : Eq a.fst b.fst
    h2 : Eq ((PiTensorProduct.reindex R (fun x => M) (finCongr h)) a.snd) b.snd
    ⊢ Eq ((PiTensorProduct.reindex R (fun x => M) (Equiv.cast ⋯)) a.snd) b.snd
  -/
  rw [← finCongr_eq_equivCast, ← h2]
  /-
    🎉 no goals
  -/


theorem cast_eq_cast {i j} (h : i = j) :
    ⇑(cast R M h) = _root_.cast (congrArg (fun i => ⨂[R]^i M) h) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i j : Nat
    h : Eq i j
    ⊢ Eq (⇑(TensorPower.cast R M h)) (_root_.cast ⋯)
  -/
  subst h
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i : Nat
    ⊢ Eq (⇑(TensorPower.cast R M ⋯)) (_root_.cast ⋯)
  -/
  rw [cast_refl]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i : Nat
    ⊢ Eq (⇑(LinearEquiv.refl R (TensorPower R i M))) (_root_.cast ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem tprod_mul_tprod {na nb} (a : Fin na → M) (b : Fin nb → M) :
    tprod R a ₜ* tprod R b = tprod R (Fin.append a b) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb : Nat
    a : Fin na → M
    b : Fin nb → M
    ⊢ Eq (GradedMonoid.GMul.mul ((PiTensorProduct.tprod R) a) ((PiTensorProduct.tp …
  -/
  dsimp [gMul_def, mulEquiv]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb : Nat
    a : Fin na → M
    b : Fin nb → M
    ⊢ Eq ((PiTensorProduct.reindex R (fun x => M) finSumFinEquiv) ((PiTensorProduc …
  -/
  rw [tmulEquiv_apply R M a b]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb : Nat
    a : Fin na → M
    b : Fin nb → M
    ⊢ Eq ((PiTensorProduct.reindex R (fun x => M) finSumFinEquiv) ((PiTensorProduc …
  -/
  refine (reindex_tprod _ _).trans ?_
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb : Nat
    a : Fin na → M
    b : Fin nb → M
    ⊢ Eq ((PiTensorProduct.tprod R) fun i => Sum.elim a b (finSumFinEquiv.symm i)) …
  -/
  congr 1
  /-
    case h.e_6.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb : Nat
    a : Fin na → M
    b : Fin nb → M
    ⊢ Eq (fun i => Sum.elim a b (finSumFinEquiv.symm i)) (Fin.append a b)
  -/
  dsimp only [Fin.append, finSumFinEquiv, Equiv.coe_fn_symm_mk]
  /-
    case h.e_6.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb : Nat
    a : Fin na → M
    b : Fin nb → M
    ⊢ Eq (fun i => Sum.elim a b (Fin.addCases Sum.inl Sum.inr i)) (Fin.addCases a b)
  -/
  apply funext
  /-
    case h.e_6.h.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb : Nat
    a : Fin na → M
    b : Fin nb → M
    ⊢ ∀ (x : Fin (HAdd.hAdd na nb)), Eq (Sum.elim a b (Fin.addCases Sum.inl Sum.in …
  -/
                         /-
                           🎉 no goals
                         -/
  apply Fin.addCases <;> simp
                         /-
                           🎉 no goals
                         -/


theorem one_mul {n} (a : ⨂[R]^n M) : cast R M (zero_add n) (ₜ1 ₜ* a) = a := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n : Nat
    a : TensorPower R n M
    ⊢ Eq ((TensorPower.cast R M ⋯) (GradedMonoid.GMul.mul GradedMonoid.GOne.one a) …
  -/
  rw [gMul_def, gOne_def]
  induction a using PiTensorProduct.induction_on with
  | smul_tprod r a =>
    rw [TensorProduct.tmul_smul, LinearEquiv.map_smul, LinearEquiv.map_smul, ← gMul_def,
      tprod_mul_tprod, cast_tprod]
    congr 2 with i
    rw [Fin.elim0_append]
    refine congr_arg a (Fin.ext ?_)
    simp
  | add x y hx hy =>
    rw [TensorProduct.tmul_add, map_add, map_add, hx, hy]


theorem mul_one {n} (a : ⨂[R]^n M) : cast R M (add_zero _) (a ₜ* ₜ1) = a := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n : Nat
    a : TensorPower R n M
    ⊢ Eq ((TensorPower.cast R M ⋯) (GradedMonoid.GMul.mul a GradedMonoid.GOne.one) …
  -/
  rw [gMul_def, gOne_def]
  induction a using PiTensorProduct.induction_on with
  | smul_tprod r a =>
    rw [← TensorProduct.smul_tmul', LinearEquiv.map_smul, LinearEquiv.map_smul, ← gMul_def,
      tprod_mul_tprod R a _, cast_tprod]
    congr 2 with i
    rw [Fin.append_elim0]
    refine congr_arg a (Fin.ext ?_)
    simp
  | add x y hx hy =>
    rw [TensorProduct.add_tmul, map_add, map_add, hx, hy]


theorem mul_assoc {na nb nc} (a : (⨂[R]^na) M) (b : (⨂[R]^nb) M) (c : (⨂[R]^nc) M) :
    cast R M (add_assoc _ _ _) (a ₜ* b ₜ* c) = a ₜ* (b ₜ* c) := by
  let mul : ∀ n m : ℕ, ⨂[R]^n M →ₗ[R] (⨂[R]^m) M →ₗ[R] (⨂[R]^(n + m)) M := fun n m =>
    (TensorProduct.mk R _ _).compr₂ ↑(mulEquiv : _ ≃ₗ[R] (⨂[R]^(n + m)) M)
  -- replace `a`, `b`, `c` with `tprod R a`, `tprod R b`, `tprod R c`
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb nc : Nat
    a : TensorPower R na M
    b : TensorPower R nb M
    c : TensorPower R nc M
    mul : (n m : Nat) → LinearMap (RingHom.id R) (TensorPower R n M) (LinearMap (R …
    ⊢ Eq ((TensorPower.cast R M ⋯) (GradedMonoid.GMul.mul (GradedMonoid.GMul.mul a …
  -/
  let e : (⨂[R]^(na + nb + nc)) M ≃ₗ[R] (⨂[R]^(na + (nb + nc))) M := cast R M (add_assoc _ _ _)
  let lhs : (⨂[R]^na) M →ₗ[R] (⨂[R]^nb) M →ₗ[R] (⨂[R]^nc) M →ₗ[R] (⨂[R]^(na + (nb + nc))) M :=
    (LinearMap.llcomp R _ _ _ ((mul _ nc).compr₂ e.toLinearMap)).comp (mul na nb)
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb nc : Nat
    a : TensorPower R na M
    b : TensorPower R nb M
    c : TensorPower R nc M
    mul : (n m : Nat) → LinearMap (RingHom.id R) (TensorPower R n M) (LinearMap (R …
    e : LinearEquiv (RingHom.id R) (TensorPower R (HAdd.hAdd (HAdd.hAdd na nb) nc) …
    lhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    ⊢ Eq ((TensorPower.cast R M ⋯) (GradedMonoid.GMul.mul (GradedMonoid.GMul.mul a …
  -/
  have lhs_eq : ∀ a b c, lhs a b c = e (a ₜ* b ₜ* c) := fun _ _ _ => rfl
  let rhs : (⨂[R]^na) M →ₗ[R] (⨂[R]^nb) M →ₗ[R] (⨂[R]^nc) M →ₗ[R] (⨂[R]^(na + (nb + nc))) M :=
    (LinearMap.llcomp R _ _ _ (LinearMap.lflip (R := R)) <|
        (LinearMap.llcomp R _ _ _ (mul na _).flip).comp (mul nb nc)).flip
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb nc : Nat
    a : TensorPower R na M
    b : TensorPower R nb M
    c : TensorPower R nc M
    mul : (n m : Nat) → LinearMap (RingHom.id R) (TensorPower R n M) (LinearMap (R …
    e : LinearEquiv (RingHom.id R) (TensorPower R (HAdd.hAdd (HAdd.hAdd na nb) nc) …
    lhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    lhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    rhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    ⊢ Eq ((TensorPower.cast R M ⋯) (GradedMonoid.GMul.mul (GradedMonoid.GMul.mul a …
  -/
  have rhs_eq : ∀ a b c, rhs a b c = a ₜ* (b ₜ* c) := fun _ _ _ => rfl
  suffices lhs = rhs from
    LinearMap.congr_fun (LinearMap.congr_fun (LinearMap.congr_fun this a) b) c
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb nc : Nat
    a : TensorPower R na M
    b : TensorPower R nb M
    c : TensorPower R nc M
    mul : (n m : Nat) → LinearMap (RingHom.id R) (TensorPower R n M) (LinearMap (R …
    e : LinearEquiv (RingHom.id R) (TensorPower R (HAdd.hAdd (HAdd.hAdd na nb) nc) …
    lhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    lhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    rhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    rhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    ⊢ Eq lhs rhs
  -/
  ext a b c
  -- clean up
  /-
    case H.H.H.H.H.H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb nc : Nat
    a✝ : TensorPower R na M
    b✝ : TensorPower R nb M
    c✝ : TensorPower R nc M
    mul : (n m : Nat) → LinearMap (RingHom.id R) (TensorPower R n M) (LinearMap (R …
    e : LinearEquiv (RingHom.id R) (TensorPower R (HAdd.hAdd (HAdd.hAdd na nb) nc) …
    lhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    lhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    rhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    rhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    a : Fin na → M
    b : Fin nb → M
    c : Fin nc → M
    ⊢ Eq ((((((lhs.compMultilinearMap (PiTensorProduct.tprod R)) a).compMultilinea …
  -/
  simp only [e, LinearMap.compMultilinearMap_apply, lhs_eq, rhs_eq, tprod_mul_tprod, cast_tprod]
  /-
    case H.H.H.H.H.H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb nc : Nat
    a✝ : TensorPower R na M
    b✝ : TensorPower R nb M
    c✝ : TensorPower R nc M
    mul : (n m : Nat) → LinearMap (RingHom.id R) (TensorPower R n M) (LinearMap (R …
    e : LinearEquiv (RingHom.id R) (TensorPower R (HAdd.hAdd (HAdd.hAdd na nb) nc) …
    lhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    lhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    rhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    rhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    a : Fin na → M
    b : Fin nb → M
    c : Fin nc → M
    ⊢ Eq ((PiTensorProduct.tprod R) (Function.comp (Fin.append (Fin.append a b) c) …
  -/
  congr with j
  /-
    case H.H.H.H.H.H.h.e_6.h.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb nc : Nat
    a✝ : TensorPower R na M
    b✝ : TensorPower R nb M
    c✝ : TensorPower R nc M
    mul : (n m : Nat) → LinearMap (RingHom.id R) (TensorPower R n M) (LinearMap (R …
    e : LinearEquiv (RingHom.id R) (TensorPower R (HAdd.hAdd (HAdd.hAdd na nb) nc) …
    lhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    lhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    rhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    rhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    a : Fin na → M
    b : Fin nb → M
    c : Fin nc → M
    j : Fin (HAdd.hAdd na (HAdd.hAdd nb nc))
    ⊢ Eq (Function.comp (Fin.append (Fin.append a b) c) (Fin.cast ⋯) j) (Fin.appen …
  -/
  rw [Fin.append_assoc]
  /-
    case H.H.H.H.H.H.h.e_6.h.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb nc : Nat
    a✝ : TensorPower R na M
    b✝ : TensorPower R nb M
    c✝ : TensorPower R nc M
    mul : (n m : Nat) → LinearMap (RingHom.id R) (TensorPower R n M) (LinearMap (R …
    e : LinearEquiv (RingHom.id R) (TensorPower R (HAdd.hAdd (HAdd.hAdd na nb) nc) …
    lhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    lhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    rhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    rhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    a : Fin na → M
    b : Fin nb → M
    c : Fin nc → M
    j : Fin (HAdd.hAdd na (HAdd.hAdd nb nc))
    ⊢ Eq (Function.comp (Function.comp (Fin.append a (Fin.append b c)) (Fin.cast ⋯ …
  -/
  refine congr_arg (Fin.append a (Fin.append b c)) (Fin.ext ?_)
  /-
    case H.H.H.H.H.H.h.e_6.h.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    na nb nc : Nat
    a✝ : TensorPower R na M
    b✝ : TensorPower R nb M
    c✝ : TensorPower R nc M
    mul : (n m : Nat) → LinearMap (RingHom.id R) (TensorPower R n M) (LinearMap (R …
    e : LinearEquiv (RingHom.id R) (TensorPower R (HAdd.hAdd (HAdd.hAdd na nb) nc) …
    lhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    lhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    rhs : LinearMap (RingHom.id R) (TensorPower R na M) (LinearMap (RingHom.id R)  …
    rhs_eq : ∀ (a : TensorPower R na M) (b : TensorPower R nb M) (c : TensorPower  …
    a : Fin na → M
    b : Fin nb → M
    c : Fin nc → M
    j : Fin (HAdd.hAdd na (HAdd.hAdd nb nc))
    ⊢ Eq ↑(Fin.cast ⋯ (Fin.cast ⋯ j)) ↑j
  -/
  rw [Fin.coe_cast, Fin.coe_cast]
  /-
    🎉 no goals
  -/

-- for now we just use the default for the `gnpow` field as it's easier.

instance gmonoid : GradedMonoid.GMonoid fun i => ⨂[R]^i M :=
  { TensorPower.gMul, TensorPower.gOne with
    one_mul := fun _ => gradedMonoid_eq_of_cast (zero_add _) (one_mul _)
    mul_one := fun _ => gradedMonoid_eq_of_cast (add_zero _) (mul_one _)
    mul_assoc := fun _ _ _ => gradedMonoid_eq_of_cast (add_assoc _ _ _) (mul_assoc _ _ _) }


/-- The canonical map from `R` to `⨂[R]^0 M` corresponding to the `algebraMap` of the tensor
algebra. -/
def algebraMap₀ : R ≃ₗ[R] (⨂[R]^0) M :=
  LinearEquiv.symm <| isEmptyEquiv (Fin 0)


theorem algebraMap₀_eq_smul_one (r : R) : (algebraMap₀ r : (⨂[R]^0) M) = r • ₜ1 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    r : R
    ⊢ Eq (TensorPower.algebraMap₀ r) (HSMul.hSMul r GradedMonoid.GOne.one)
  -/
  simp [algebraMap₀]; congr
                      /-
                        🎉 no goals
                      -/


theorem algebraMap₀_one : (algebraMap₀ 1 : (⨂[R]^0) M) = ₜ1 :=
  (algebraMap₀_eq_smul_one 1).trans (one_smul _ _)


theorem algebraMap₀_mul {n} (r : R) (a : ⨂[R]^n M) :
    cast R M (zero_add _) (algebraMap₀ r ₜ* a) = r • a := by
  rw [gMul_eq_coe_linearMap, algebraMap₀_eq_smul_one, LinearMap.map_smul₂,
    LinearEquiv.map_smul, ← gMul_eq_coe_linearMap, one_mul]


theorem mul_algebraMap₀ {n} (r : R) (a : ⨂[R]^n M) :
    cast R M (add_zero _) (a ₜ* algebraMap₀ r) = r • a := by
  rw [gMul_eq_coe_linearMap, algebraMap₀_eq_smul_one, LinearMap.map_smul,
    LinearEquiv.map_smul, ← gMul_eq_coe_linearMap, mul_one]


theorem algebraMap₀_mul_algebraMap₀ (r s : R) :
    cast R M (add_zero _) (algebraMap₀ r ₜ* algebraMap₀ s) = algebraMap₀ (r * s) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    r s : R
    ⊢ Eq ((TensorPower.cast R M ⋯) (GradedMonoid.GMul.mul (TensorPower.algebraMap₀ …
  -/
  rw [← smul_eq_mul, LinearEquiv.map_smul]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    r s : R
    ⊢ Eq ((TensorPower.cast R M ⋯) (GradedMonoid.GMul.mul (TensorPower.algebraMap₀ …
  -/
  exact algebraMap₀_mul r (@algebraMap₀ R M _ _ _ s)
  /-
    🎉 no goals
  -/


instance gsemiring : DirectSum.GSemiring fun i => ⨂[R]^i M :=
  { TensorPower.gmonoid with
    mul_zero := fun _ => LinearMap.map_zero _
    zero_mul := fun _ => LinearMap.map_zero₂ _ _
    mul_add := fun _ _ _ => LinearMap.map_add _ _ _
    add_mul := fun _ _ _ => LinearMap.map_add₂ _ _ _ _
    natCast := fun n => algebraMap₀ (n : R)
                       /-
                         R : Type u_1
                         M : Type u_2
                         inst✝² : CommSemiring R
                         inst✝¹ : AddCommMonoid M
                         inst✝ : Module R M
                         ⊢ Eq ((fun n => TensorPower.algebraMap₀ ↑n) 0) 0
                       -/
    natCast_zero := by simp only [Nat.cast_zero, map_zero]
                       /-
                         🎉 no goals
                       -/
                                /-
                                  R : Type u_1
                                  M : Type u_2
                                  inst✝² : CommSemiring R
                                  inst✝¹ : AddCommMonoid M
                                  inst✝ : Module R M
                                  n : Nat
                                  ⊢ Eq ((fun n => TensorPower.algebraMap₀ ↑n) (HAdd.hAdd n 1)) (HAdd.hAdd ((fun  …
                                -/
    natCast_succ := fun n => by simp only [Nat.cast_succ, map_add, algebraMap₀_one] }
                                /-
                                  🎉 no goals
                                -/


/-- The tensor powers form a graded algebra.

Note that this instance implies `Algebra R (⨁ n : ℕ, ⨂[R]^n M)` via `DirectSum.Algebra`. -/
instance galgebra : DirectSum.GAlgebra R fun i => ⨂[R]^i M where
  toFun := (algebraMap₀ : R ≃ₗ[R] (⨂[R]^0) M).toLinearMap.toAddMonoidHom
  map_one := algebraMap₀_one
  map_mul r s := gradedMonoid_eq_of_cast rfl (by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r s : R
      ⊢ Eq ((TensorPower.cast R M ⋯) (GradedMonoid.mk 0 ((↑TensorPower.algebraMap₀). …
    -/
    rw [← LinearEquiv.eq_symm_apply]
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r s : R
      ⊢ Eq (GradedMonoid.mk 0 ((↑TensorPower.algebraMap₀).toAddMonoidHom (HMul.hMul  …
    -/
    have := algebraMap₀_mul_algebraMap₀ (M := M) r s
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r s : R
      this : Eq ((TensorPower.cast R M ⋯) (GradedMonoid.GMul.mul (TensorPower.algebr …
      ⊢ Eq (GradedMonoid.mk 0 ((↑TensorPower.algebraMap₀).toAddMonoidHom (HMul.hMul  …
    -/
    exact this.symm)
    /-
      🎉 no goals
    -/
  commutes r x := gradedMonoid_eq_of_cast (add_comm _ _) (by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r : R
      x : GradedMonoid fun i => TensorPower R i M
      ⊢ Eq ((TensorPower.cast R M ⋯) (HMul.hMul (GradedMonoid.mk 0 ((↑TensorPower.al …
    -/
    have := (algebraMap₀_mul r x.snd).trans (mul_algebraMap₀ r x.snd).symm
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r : R
      x : GradedMonoid fun i => TensorPower R i M
      this : Eq ((TensorPower.cast R M ⋯) (GradedMonoid.GMul.mul (TensorPower.algebr …
      ⊢ Eq ((TensorPower.cast R M ⋯) (HMul.hMul (GradedMonoid.mk 0 ((↑TensorPower.al …
    -/
    rw [← LinearEquiv.eq_symm_apply, cast_symm]
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r : R
      x : GradedMonoid fun i => TensorPower R i M
      this : Eq ((TensorPower.cast R M ⋯) (GradedMonoid.GMul.mul (TensorPower.algebr …
      ⊢ Eq (HMul.hMul (GradedMonoid.mk 0 ((↑TensorPower.algebraMap₀).toAddMonoidHom  …
    -/
    rw [← LinearEquiv.eq_symm_apply, cast_symm, cast_cast] at this
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r : R
      x : GradedMonoid fun i => TensorPower R i M
      this : Eq (GradedMonoid.GMul.mul (TensorPower.algebraMap₀ r) x.snd) ((TensorPo …
      ⊢ Eq (HMul.hMul (GradedMonoid.mk 0 ((↑TensorPower.algebraMap₀).toAddMonoidHom  …
    -/
    exact this)
    /-
      🎉 no goals
    -/
  smul_def r x := gradedMonoid_eq_of_cast (zero_add x.fst).symm (by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r : R
      x : GradedMonoid fun i => TensorPower R i M
      ⊢ Eq ((TensorPower.cast R M ⋯) (HSMul.hSMul r x).snd) (HMul.hMul (GradedMonoid …
    -/
    rw [← LinearEquiv.eq_symm_apply, cast_symm]
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      r : R
      x : GradedMonoid fun i => TensorPower R i M
      ⊢ Eq (HSMul.hSMul r x).snd ((TensorPower.cast R M ⋯) (HMul.hMul (GradedMonoid. …
    -/
    exact (algebraMap₀_mul r x.snd).symm)
    /-
      🎉 no goals
    -/


theorem galgebra_toFun_def (r : R) :
    DirectSum.GAlgebra.toFun (A := fun i ↦ ⨂[R]^i M) r = algebraMap₀ r :=
  rfl


