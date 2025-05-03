/-- The dual space of an R-module M is the R-module of linear maps `M → R`. -/
abbrev Dual :=
  M →ₗ[R] R


/-- The canonical pairing of a vector space and its algebraic dual. -/
def dualPairing (R M) [CommSemiring R] [AddCommMonoid M] [Module R M] :
    Module.Dual R M →ₗ[R] M →ₗ[R] R :=
  LinearMap.id


@[simp]
theorem dualPairing_apply (v x) : dualPairing R M v x = v x :=
  rfl


instance : Inhabited (Dual R M) := ⟨0⟩


/-- Maps a module M to the dual of the dual of M. See `Module.erange_coe` and
`Module.evalEquiv`. -/
def eval : M →ₗ[R] Dual R (Dual R M) :=
  LinearMap.flip LinearMap.id


@[simp]
theorem eval_apply (v : M) (a : Dual R M) : eval R M v a = a v :=
  rfl


/-- The transposition of linear maps, as a linear map from `M →ₗ[R] M'` to
`Dual R M' →ₗ[R] Dual R M`. -/
def transpose : (M →ₗ[R] M') →ₗ[R] Dual R M' →ₗ[R] Dual R M :=
  (LinearMap.llcomp R M M' R).flip

-- Porting note: with reducible def need to specify some parameters to transpose explicitly

theorem transpose_apply (u : M →ₗ[R] M') (l : Dual R M') : transpose (R := R) u l = l.comp u :=
  rfl


theorem transpose_comp (u : M' →ₗ[R] M'') (v : M →ₗ[R] M') :
    transpose (R := R) (u.comp v) = (transpose (R := R) v).comp (transpose (R := R) u) :=
  rfl


/-- Taking duals distributes over products. -/
@[simps!]
def dualProdDualEquivDual : (Module.Dual R M × Module.Dual R M') ≃ₗ[R] Module.Dual R (M × M') :=
  LinearMap.coprodEquiv R


@[simp]
theorem dualProdDualEquivDual_apply (φ : Module.Dual R M) (ψ : Module.Dual R M') :
    dualProdDualEquivDual R M M' (φ, ψ) = φ.coprod ψ :=
  rfl


/-- Given a linear map `f : M₁ →ₗ[R] M₂`, `f.dualMap` is the linear map between the dual of
`M₂` and `M₁` such that it maps the functional `φ` to `φ ∘ f`. -/
def LinearMap.dualMap (f : M₁ →ₗ[R] M₂) : Dual R M₂ →ₗ[R] Dual R M₁ :=
-- Porting note: with reducible def need to specify some parameters to transpose explicitly
  Module.Dual.transpose (R := R) f


lemma LinearMap.dualMap_eq_lcomp (f : M₁ →ₗ[R] M₂) : f.dualMap = f.lcomp R R := rfl

-- Porting note: with reducible def need to specify some parameters to transpose explicitly

theorem LinearMap.dualMap_def (f : M₁ →ₗ[R] M₂) : f.dualMap = Module.Dual.transpose (R := R) f :=
  rfl


theorem LinearMap.dualMap_apply' (f : M₁ →ₗ[R] M₂) (g : Dual R M₂) : f.dualMap g = g.comp f :=
  rfl


@[simp]
theorem LinearMap.dualMap_apply (f : M₁ →ₗ[R] M₂) (g : Dual R M₂) (x : M₁) :
    f.dualMap g x = g (f x) :=
  rfl


@[simp]
theorem LinearMap.dualMap_id : (LinearMap.id : M₁ →ₗ[R] M₁).dualMap = LinearMap.id := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    M₁ : Type v
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    ⊢ Eq LinearMap.id.dualMap LinearMap.id
  -/
  ext
  /-
    case h.h
    R : Type u
    inst✝² : CommSemiring R
    M₁ : Type v
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    x✝¹ : Module.Dual R M₁
    x✝ : M₁
    ⊢ Eq ((LinearMap.id.dualMap x✝¹) x✝) ((LinearMap.id x✝¹) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem LinearMap.dualMap_comp_dualMap {M₃ : Type*} [AddCommGroup M₃] [Module R M₃]
    (f : M₁ →ₗ[R] M₂) (g : M₂ →ₗ[R] M₃) : f.dualMap.comp g.dualMap = (g.comp f).dualMap :=
  rfl


/-- If a linear map is surjective, then its dual is injective. -/
theorem LinearMap.dualMap_injective_of_surjective {f : M₁ →ₗ[R] M₂} (hf : Function.Surjective f) :
    Function.Injective f.dualMap := by
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    M₁ : Type v
    M₂ : Type v'
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    hf : Function.Surjective ⇑f
    ⊢ Function.Injective ⇑f.dualMap
  -/
  intro φ ψ h
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    M₁ : Type v
    M₂ : Type v'
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    hf : Function.Surjective ⇑f
    φ ψ : Module.Dual R M₂
    h : Eq (f.dualMap φ) (f.dualMap ψ)
    ⊢ Eq φ ψ
  -/
  ext x
  /-
    case h
    R : Type u
    inst✝⁴ : CommSemiring R
    M₁ : Type v
    M₂ : Type v'
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    hf : Function.Surjective ⇑f
    φ ψ : Module.Dual R M₂
    h : Eq (f.dualMap φ) (f.dualMap ψ)
    x : M₂
    ⊢ Eq (φ x) (ψ x)
  -/
  obtain ⟨y, rfl⟩ := hf x
  /-
    case h.intro
    R : Type u
    inst✝⁴ : CommSemiring R
    M₁ : Type v
    M₂ : Type v'
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    hf : Function.Surjective ⇑f
    φ ψ : Module.Dual R M₂
    h : Eq (f.dualMap φ) (f.dualMap ψ)
    y : M₁
    ⊢ Eq (φ (f y)) (ψ (f y))
  -/
  exact congr_arg (fun g : Module.Dual R M₁ => g y) h
  /-
    🎉 no goals
  -/


/-- The `Linear_equiv` version of `LinearMap.dualMap`. -/
def LinearEquiv.dualMap (f : M₁ ≃ₗ[R] M₂) : Dual R M₂ ≃ₗ[R] Dual R M₁ where
  __ := f.toLinearMap.dualMap
  invFun := f.symm.toLinearMap.dualMap
  left_inv φ := LinearMap.ext fun x ↦ congr_arg φ (f.right_inv x)
  right_inv φ := LinearMap.ext fun x ↦ congr_arg φ (f.left_inv x)


@[simp]
theorem LinearEquiv.dualMap_apply (f : M₁ ≃ₗ[R] M₂) (g : Dual R M₂) (x : M₁) :
    f.dualMap g x = g (f x) :=
  rfl


@[simp]
theorem LinearEquiv.dualMap_refl :
    (LinearEquiv.refl R M₁).dualMap = LinearEquiv.refl R (Dual R M₁) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    M₁ : Type v
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    ⊢ Eq (LinearEquiv.refl R M₁).dualMap (LinearEquiv.refl R (Module.Dual R M₁))
  -/
  ext
  /-
    case h.h
    R : Type u
    inst✝² : CommSemiring R
    M₁ : Type v
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    x✝¹ : Module.Dual R M₁
    x✝ : M₁
    ⊢ Eq (((LinearEquiv.refl R M₁).dualMap x✝¹) x✝) (((LinearEquiv.refl R (Module. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem LinearEquiv.dualMap_symm {f : M₁ ≃ₗ[R] M₂} :
    (LinearEquiv.dualMap f).symm = LinearEquiv.dualMap f.symm :=
  rfl


theorem LinearEquiv.dualMap_trans {M₃ : Type*} [AddCommGroup M₃] [Module R M₃] (f : M₁ ≃ₗ[R] M₂)
    (g : M₂ ≃ₗ[R] M₃) : g.dualMap.trans f.dualMap = (f.trans g).dualMap :=
  rfl


theorem Module.Dual.eval_naturality (f : M₁ →ₗ[R] M₂) :
    f.dualMap.dualMap ∘ₗ eval R M₁ = eval R M₂ ∘ₗ f := by
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    M₁ : Type v
    M₂ : Type v'
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    ⊢ Eq (f.dualMap.dualMap.comp (Module.Dual.eval R M₁)) ((Module.Dual.eval R M₂) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma Dual.apply_one_mul_eq (f : Dual R R) (r : R) :
    f 1 * r = f r := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    f : Module.Dual R R
    r : R
    ⊢ Eq (HMul.hMul (f 1) r) (f r)
  -/
  conv_rhs => rw [← mul_one r, ← smul_eq_mul]
  /-
    R : Type u
    inst✝ : CommSemiring R
    f : Module.Dual R R
    r : R
    ⊢ Eq (HMul.hMul (f 1) r) (f (HSMul.hSMul r 1))
  -/
  rw [map_smul, smul_eq_mul, mul_comm]
  /-
    🎉 no goals
  -/


@[simp]
lemma LinearMap.range_dualMap_dual_eq_span_singleton (f : Dual R M₁) :
    range f.dualMap = R ∙ f := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    M₁ : Type v
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    f : Module.Dual R M₁
    ⊢ Eq (LinearMap.range (LinearMap.dualMap f)) (Submodule.span R (Singleton.sing …
  -/
  ext m
  /-
    case h
    R : Type u
    inst✝² : CommSemiring R
    M₁ : Type v
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    f m : Module.Dual R M₁
    ⊢ Iff (Membership.mem (LinearMap.range (LinearMap.dualMap f)) m) (Membership.m …
  -/
  rw [Submodule.mem_span_singleton]
  /-
    case h
    R : Type u
    inst✝² : CommSemiring R
    M₁ : Type v
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    f m : Module.Dual R M₁
    ⊢ Iff (Membership.mem (LinearMap.range (LinearMap.dualMap f)) m) (Exists fun a …
  -/
  refine ⟨fun ⟨r, hr⟩ ↦ ⟨r 1, ?_⟩, fun ⟨r, hr⟩ ↦ ⟨r • LinearMap.id, ?_⟩⟩
    /-
      case h.refine_1
      R : Type u
      inst✝² : CommSemiring R
      M₁ : Type v
      inst✝¹ : AddCommMonoid M₁
      inst✝ : Module R M₁
      f m : Module.Dual R M₁
      x✝ : Membership.mem (LinearMap.range (LinearMap.dualMap f)) m
      r : Module.Dual R R
      hr : Eq ((LinearMap.dualMap f) r) m
      ⊢ Eq (HSMul.hSMul (r 1) f) m
    -/
  · ext; simp [dualMap_apply', ← hr]
         /-
           🎉 no goals
         -/
    /-
      case h.refine_2
      R : Type u
      inst✝² : CommSemiring R
      M₁ : Type v
      inst✝¹ : AddCommMonoid M₁
      inst✝ : Module R M₁
      f m : Module.Dual R M₁
      x✝ : Exists fun a => Eq (HSMul.hSMul a f) m
      r : R
      hr : Eq (HSMul.hSMul r f) m
      ⊢ Eq ((LinearMap.dualMap f) (HSMul.hSMul r LinearMap.id)) m
    -/
  · ext; simp [dualMap_apply', ← hr]
         /-
           🎉 no goals
         -/


/-- The linear map from a vector space equipped with basis to its dual vector space,
taking basis elements to corresponding dual basis elements. -/
def toDual : M →ₗ[R] Module.Dual R M :=
  b.constr ℕ fun v => b.constr ℕ fun w => if w = v then (1 : R) else 0


theorem toDual_apply (i j : ι) : b.toDual (b i) (b j) = if i = j then 1 else 0 := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq ι
    b : Basis ι R M
    i j : ι
    ⊢ Eq ((b.toDual (b i)) (b j)) (ite (Eq i j) 1 0)
  -/
  erw [constr_basis b, constr_basis b]
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq ι
    b : Basis ι R M
    i j : ι
    ⊢ Eq (ite (Eq j i) 1 0) (ite (Eq i j) 1 0)
  -/
  simp only [eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem toDual_linearCombination_left (f : ι →₀ R) (i : ι) :
    b.toDual (Finsupp.linearCombination R b f) (b i) = f i := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq ι
    b : Basis ι R M
    f : Finsupp ι R
    i : ι
    ⊢ Eq ((b.toDual ((Finsupp.linearCombination R ⇑b) f)) (b i)) (f i)
  -/
  rw [Finsupp.linearCombination_apply, Finsupp.sum, _root_.map_sum, LinearMap.sum_apply]
  simp_rw [LinearMap.map_smul, LinearMap.smul_apply, toDual_apply, smul_eq_mul, mul_boole,
    Finset.sum_ite_eq']
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq ι
    b : Basis ι R M
    f : Finsupp ι R
    i : ι
    ⊢ Eq (ite (Membership.mem f.support i) (f i) 0) (f i)
  -/
  split_ifs with h
    /-
      case pos
      R : Type uR
      M : Type uM
      ι : Type uι
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      inst✝ : DecidableEq ι
      b : Basis ι R M
      f : Finsupp ι R
      i : ι
      h : Membership.mem f.support i
      ⊢ Eq (f i) (f i)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type uR
      M : Type uM
      ι : Type uι
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      inst✝ : DecidableEq ι
      b : Basis ι R M
      f : Finsupp ι R
      i : ι
      h : Not (Membership.mem f.support i)
      ⊢ Eq 0 (f i)
    -/
  · rw [Finsupp.not_mem_support_iff.mp h]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-08-29")] alias toDual_total_left := toDual_linearCombination_left


@[simp]
theorem toDual_linearCombination_right (f : ι →₀ R) (i : ι) :
    b.toDual (b i) (Finsupp.linearCombination R b f) = f i := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq ι
    b : Basis ι R M
    f : Finsupp ι R
    i : ι
    ⊢ Eq ((b.toDual (b i)) ((Finsupp.linearCombination R ⇑b) f)) (f i)
  -/
  rw [Finsupp.linearCombination_apply, Finsupp.sum, _root_.map_sum]
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq ι
    b : Basis ι R M
    f : Finsupp ι R
    i : ι
    ⊢ Eq (f.support.sum fun x => (b.toDual (b i)) (HSMul.hSMul (f x) (b x))) (f i)
  -/
  simp_rw [LinearMap.map_smul, toDual_apply, smul_eq_mul, mul_boole, Finset.sum_ite_eq]
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq ι
    b : Basis ι R M
    f : Finsupp ι R
    i : ι
    ⊢ Eq (ite (Membership.mem f.support i) (f i) 0) (f i)
  -/
  split_ifs with h
    /-
      case pos
      R : Type uR
      M : Type uM
      ι : Type uι
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      inst✝ : DecidableEq ι
      b : Basis ι R M
      f : Finsupp ι R
      i : ι
      h : Membership.mem f.support i
      ⊢ Eq (f i) (f i)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type uR
      M : Type uM
      ι : Type uι
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      inst✝ : DecidableEq ι
      b : Basis ι R M
      f : Finsupp ι R
      i : ι
      h : Not (Membership.mem f.support i)
      ⊢ Eq 0 (f i)
    -/
  · rw [Finsupp.not_mem_support_iff.mp h]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-08-29")] alias toDual_total_right :=
  toDual_linearCombination_right


theorem toDual_apply_left (m : M) (i : ι) : b.toDual m (b i) = b.repr m i := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq ι
    b : Basis ι R M
    m : M
    i : ι
    ⊢ Eq ((b.toDual m) (b i)) ((b.repr m) i)
  -/
  rw [← b.toDual_linearCombination_left, b.linearCombination_repr]
  /-
    🎉 no goals
  -/


theorem toDual_apply_right (i : ι) (m : M) : b.toDual (b i) m = b.repr m i := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq ι
    b : Basis ι R M
    i : ι
    m : M
    ⊢ Eq ((b.toDual (b i)) m) ((b.repr m) i)
  -/
  rw [← b.toDual_linearCombination_right, b.linearCombination_repr]
  /-
    🎉 no goals
  -/


theorem coe_toDual_self (i : ι) : b.toDual (b i) = b.coord i := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq ι
    b : Basis ι R M
    i : ι
    ⊢ Eq (b.toDual (b i)) (b.coord i)
  -/
  ext
  /-
    case h
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq ι
    b : Basis ι R M
    i : ι
    x✝ : M
    ⊢ Eq ((b.toDual (b i)) x✝) ((b.coord i) x✝)
  -/
  apply toDual_apply_right
  /-
    🎉 no goals
  -/


/-- `h.toDual_flip v` is the linear map sending `w` to `h.toDual w v`. -/
def toDualFlip (m : M) : M →ₗ[R] R :=
  b.toDual.flip m


theorem toDualFlip_apply (m₁ m₂ : M) : b.toDualFlip m₁ m₂ = b.toDual m₂ m₁ :=
  rfl


theorem toDual_eq_repr (m : M) (i : ι) : b.toDual m (b i) = b.repr m i :=
  b.toDual_apply_left m i


theorem toDual_eq_equivFun [Finite ι] (m : M) (i : ι) : b.toDual m (b i) = b.equivFun m i := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    m : M
    i : ι
    ⊢ Eq ((b.toDual m) (b i)) (b.equivFun m i)
  -/
  rw [b.equivFun_apply, toDual_eq_repr]
  /-
    🎉 no goals
  -/


theorem toDual_injective : Injective b.toDual := fun x y h ↦ b.ext_elem_iff.mpr fun i ↦ by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq ι
    b : Basis ι R M
    x y : M
    h : Eq (b.toDual x) (b.toDual y)
    i : ι
    ⊢ Eq ((b.repr x) i) ((b.repr y) i)
  -/
  simp_rw [← toDual_eq_repr]; exact DFunLike.congr_fun h _
                              /-
                                🎉 no goals
                              -/


theorem toDual_inj (m : M) (a : b.toDual m = 0) : m = 0 :=
                         /-
                           R : Type uR
                           M : Type uM
                           ι : Type uι
                           inst✝³ : CommSemiring R
                           inst✝² : AddCommMonoid M
                           inst✝¹ : Module R M
                           inst✝ : DecidableEq ι
                           b : Basis ι R M
                           m : M
                           a : Eq (b.toDual m) 0
                           ⊢ Eq (b.toDual m) (b.toDual 0)
                         -/
  b.toDual_injective (by rwa [_root_.map_zero])
                         /-
                           🎉 no goals
                         -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629 LinearMap.ker

theorem toDual_ker : LinearMap.ker b.toDual = ⊥ :=
  ker_eq_bot'.mpr b.toDual_inj

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629 LinearMap.range

theorem toDual_range [Finite ι] : LinearMap.range b.toDual = ⊤ := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    ⊢ Eq (LinearMap.range b.toDual) Top.top
  -/
  refine eq_top_iff'.2 fun f => ?_
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    f : Module.Dual R M
    ⊢ Membership.mem (LinearMap.range b.toDual) f
  -/
  let lin_comb : ι →₀ R := Finsupp.equivFunOnFinite.symm fun i => f (b i)
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    f : Module.Dual R M
    lin_comb : Finsupp ι R := Finsupp.equivFunOnFinite.symm fun i => f (b i)
    ⊢ Membership.mem (LinearMap.range b.toDual) f
  -/
  refine ⟨Finsupp.linearCombination R b lin_comb, b.ext fun i => ?_⟩
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    f : Module.Dual R M
    lin_comb : Finsupp ι R := Finsupp.equivFunOnFinite.symm fun i => f (b i)
    i : ι
    ⊢ Eq ((b.toDual ((Finsupp.linearCombination R ⇑b) lin_comb)) (b i)) (f (b i))
  -/
  rw [b.toDual_eq_repr _ i, repr_linearCombination b]
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    f : Module.Dual R M
    lin_comb : Finsupp ι R := Finsupp.equivFunOnFinite.symm fun i => f (b i)
    i : ι
    ⊢ Eq (lin_comb i) (f (b i))
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem sum_dual_apply_smul_coord (f : Module.Dual R M) :
    (∑ x, f (b x) • b.coord x) = f := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Fintype ι
    b : Basis ι R M
    f : Module.Dual R M
    ⊢ Eq (Finset.univ.sum fun x => HSMul.hSMul (f (b x)) (b.coord x)) f
  -/
  ext m
  simp_rw [LinearMap.sum_apply, LinearMap.smul_apply, smul_eq_mul, mul_comm (f _), ← smul_eq_mul, ←
    f.map_smul, ← _root_.map_sum, Basis.coord_apply, Basis.sum_repr]


/-- A vector space is linearly equivalent to its dual space. -/
def toDualEquiv : M ≃ₗ[R] Dual R M :=
  .ofBijective b.toDual ⟨ker_eq_bot.mp b.toDual_ker, range_eq_top.mp b.toDual_range⟩

-- `simps` times out when generating this

@[simp]
theorem toDualEquiv_apply (m : M) : b.toDualEquiv m = b.toDual m :=
  rfl

-- Not sure whether this is true for free modules over a commutative ring

/-- A vector space over a field is isomorphic to its dual if and only if it is finite-dimensional:
  a consequence of the Erdős-Kaplansky theorem. -/
theorem linearEquiv_dual_iff_finiteDimensional [Field K] [AddCommGroup V] [Module K V] :
    Nonempty (V ≃ₗ[K] Dual K V) ↔ FiniteDimensional K V := by
  /-
    K : Type uK
    V : Type uV
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ⊢ Iff (Nonempty (LinearEquiv (RingHom.id K) V (Module.Dual K V))) (FiniteDimen …
  -/
  refine ⟨fun ⟨e⟩ ↦ ?_, fun h ↦ ⟨(Module.Free.chooseBasis K V).toDualEquiv⟩⟩
  /-
    K : Type uK
    V : Type uV
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x✝ : Nonempty (LinearEquiv (RingHom.id K) V (Module.Dual K V))
    e : LinearEquiv (RingHom.id K) V (Module.Dual K V)
    ⊢ FiniteDimensional K V
  -/
  rw [FiniteDimensional, ← Module.rank_lt_aleph0_iff]
  /-
    K : Type uK
    V : Type uV
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x✝ : Nonempty (LinearEquiv (RingHom.id K) V (Module.Dual K V))
    e : LinearEquiv (RingHom.id K) V (Module.Dual K V)
    ⊢ LT.lt (Module.rank K V) Cardinal.aleph0
  -/
  by_contra!
  /-
    K : Type uK
    V : Type uV
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x✝ : Nonempty (LinearEquiv (RingHom.id K) V (Module.Dual K V))
    e : LinearEquiv (RingHom.id K) V (Module.Dual K V)
    this : LE.le Cardinal.aleph0 (Module.rank K V)
    ⊢ False
  -/
  apply (lift_rank_lt_rank_dual this).ne
  /-
    K : Type uK
    V : Type uV
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x✝ : Nonempty (LinearEquiv (RingHom.id K) V (Module.Dual K V))
    e : LinearEquiv (RingHom.id K) V (Module.Dual K V)
    this : LE.le Cardinal.aleph0 (Module.rank K V)
    ⊢ Eq (Cardinal.lift.{uK, uV} (Module.rank K V)) (Module.rank K (LinearMap (Rin …
  -/
  have := e.lift_rank_eq
  /-
    K : Type uK
    V : Type uV
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x✝ : Nonempty (LinearEquiv (RingHom.id K) V (Module.Dual K V))
    e : LinearEquiv (RingHom.id K) V (Module.Dual K V)
    this✝ : LE.le Cardinal.aleph0 (Module.rank K V)
    this : Eq (Cardinal.lift.{max uK uV, uV} (Module.rank K V)) (Cardinal.lift.{uV …
    ⊢ Eq (Cardinal.lift.{uK, uV} (Module.rank K V)) (Module.rank K (LinearMap (Rin …
  -/
  rwa [lift_umax.{uV,uK}, lift_id'.{uV,uK}] at this
  /-
    🎉 no goals
  -/


/-- Maps a basis for `V` to a basis for the dual space. -/
def dualBasis : Basis ι R (Dual R M) :=
  b.map b.toDualEquiv

-- We use `j = i` to match `Basis.repr_self`

theorem dualBasis_apply_self (i j : ι) : b.dualBasis i (b j) =
    if j = i then 1 else 0 := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    i j : ι
    ⊢ Eq ((b.dualBasis i) (b j)) (ite (Eq j i) 1 0)
  -/
  convert b.toDual_apply i j using 2
  /-
    case h.e'_3.h₁.a
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    i j : ι
    ⊢ Iff (Eq j i) (Eq i j)
  -/
  rw [@eq_comm _ j i]
  /-
    🎉 no goals
  -/


theorem linearCombination_dualBasis (f : ι →₀ R) (i : ι) :
    Finsupp.linearCombination R b.dualBasis f (b i) = f i := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    f : Finsupp ι R
    i : ι
    ⊢ Eq (((Finsupp.linearCombination R ⇑b.dualBasis) f) (b i)) (f i)
  -/
  cases nonempty_fintype ι
  /-
    case intro
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    f : Finsupp ι R
    i : ι
    val✝ : Fintype ι
    ⊢ Eq (((Finsupp.linearCombination R ⇑b.dualBasis) f) (b i)) (f i)
  -/
  rw [Finsupp.linearCombination_apply, Finsupp.sum_fintype, LinearMap.sum_apply]
  · simp_rw [LinearMap.smul_apply, smul_eq_mul, dualBasis_apply_self, mul_boole,
    Finset.sum_ite_eq, if_pos (Finset.mem_univ i)]
    /-
      case intro.h
      R : Type uR
      M : Type uM
      ι : Type uι
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : DecidableEq ι
      b : Basis ι R M
      inst✝ : Finite ι
      f : Finsupp ι R
      i : ι
      val✝ : Fintype ι
      ⊢ ∀ (i : ι), Eq (HSMul.hSMul 0 (b.dualBasis i)) 0
    -/
  · intro
    /-
      case intro.h
      R : Type uR
      M : Type uM
      ι : Type uι
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : DecidableEq ι
      b : Basis ι R M
      inst✝ : Finite ι
      f : Finsupp ι R
      i : ι
      val✝ : Fintype ι
      i✝ : ι
      ⊢ Eq (HSMul.hSMul 0 (b.dualBasis i✝)) 0
    -/
    rw [zero_smul]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-08-29")] alias total_dualBasis := linearCombination_dualBasis


@[simp] theorem dualBasis_repr (l : Dual R M) (i : ι) : b.dualBasis.repr l i = l (b i) := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    l : Module.Dual R M
    i : ι
    ⊢ Eq ((b.dualBasis.repr l) i) (l (b i))
  -/
  rw [← linearCombination_dualBasis b, Basis.linearCombination_repr b.dualBasis l]
  /-
    🎉 no goals
  -/


theorem dualBasis_apply (i : ι) (m : M) : b.dualBasis i m = b.repr m i :=
  b.toDual_apply_right i m


@[simp]
theorem coe_dualBasis : ⇑b.dualBasis = b.coord := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    ⊢ Eq (⇑b.dualBasis) b.coord
  -/
  ext i x
  /-
    case h.h
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    i : ι
    x : M
    ⊢ Eq ((b.dualBasis i) x) ((b.coord i) x)
  -/
  apply dualBasis_apply
  /-
    🎉 no goals
  -/


@[simp]
theorem toDual_toDual : b.dualBasis.toDual.comp b.toDual = Dual.eval R M := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : DecidableEq ι
    b : Basis ι R M
    inst✝ : Finite ι
    ⊢ Eq (b.dualBasis.toDual.comp b.toDual) (Module.Dual.eval R M)
  -/
  refine b.ext fun i => b.dualBasis.ext fun j => ?_
  rw [LinearMap.comp_apply, toDual_apply_left, coe_toDual_self, ← coe_dualBasis,
    Dual.eval_apply, Basis.repr_self, Finsupp.single_apply, dualBasis_apply_self]


theorem dualBasis_equivFun [Finite ι] (l : Dual R M) (i : ι) :
                                             /-
                                               R : Type uR
                                               M : Type uM
                                               ι : Type uι
                                               inst✝⁴ : CommRing R
                                               inst✝³ : AddCommGroup M
                                               inst✝² : Module R M
                                               inst✝¹ : DecidableEq ι
                                               b : Basis ι R M
                                               inst✝ : Finite ι
                                               l : Module.Dual R M
                                               i : ι
                                               ⊢ Eq (b.dualBasis.equivFun l i) (l (b i))
                                             -/
    b.dualBasis.equivFun l i = l (b i) := by rw [Basis.equivFun_apply, dualBasis_repr]
                                             /-
                                               🎉 no goals
                                             -/


theorem eval_ker {ι : Type*} (b : Basis ι R M) :
    LinearMap.ker (Dual.eval R M) = ⊥ := by
  /-
    R : Type uR
    M : Type uM
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_1
    b : Basis ι R M
    ⊢ Eq (LinearMap.ker (Module.Dual.eval R M)) Bot.bot
  -/
  rw [ker_eq_bot']
  /-
    R : Type uR
    M : Type uM
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_1
    b : Basis ι R M
    ⊢ ∀ (m : M), Eq ((Module.Dual.eval R M) m) 0 → Eq m 0
  -/
  intro m hm
  /-
    R : Type uR
    M : Type uM
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_1
    b : Basis ι R M
    m : M
    hm : Eq ((Module.Dual.eval R M) m) 0
    ⊢ Eq m 0
  -/
  simp_rw [LinearMap.ext_iff, Dual.eval_apply, zero_apply] at hm
  /-
    R : Type uR
    M : Type uM
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_1
    b : Basis ι R M
    m : M
    hm : ∀ (x : Module.Dual R M), Eq (x m) 0
    ⊢ Eq m 0
  -/
  exact (Basis.forall_coord_eq_zero_iff _).mp fun i => hm (b.coord i)
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629 LinearMap.range

theorem eval_range {ι : Type*} [Finite ι] (b : Basis ι R M) :
    LinearMap.range (Dual.eval R M) = ⊤ := by
  classical
    cases nonempty_fintype ι
    rw [← b.toDual_toDual, range_comp, b.toDual_range, Submodule.map_top, toDual_range _]


instance dual_free [Free R M] : Free R (Dual R M) :=
  Free.of_basis (Free.chooseBasis R M).dualBasis


instance dual_projective [Projective R M] : Projective R (Dual R M) :=
  have ⟨_, f, g, _, _, hfg⟩ := Finite.exists_comp_eq_id_of_projective R M
  .of_split f.dualMap g.dualMap (congr_arg dualMap hfg)


instance dual_finite [Projective R M] : Module.Finite R (Dual R M) :=
  have ⟨n, f, g, _, _, hfg⟩ := Finite.exists_comp_eq_id_of_projective R M
  have := Finite.of_basis (Free.chooseBasis R <| Fin n → R).dualBasis
  .of_surjective _ (surjective_of_comp_eq_id f.dualMap g.dualMap <| congr_arg dualMap hfg)


/-- `simp` normal form version of `linearCombination_dualBasis` -/
@[simp]
theorem linearCombination_coord [CommRing R] [AddCommGroup M] [Module R M] [Finite ι]
    (b : Basis ι R M) (f : ι →₀ R) (i : ι) : Finsupp.linearCombination R b.coord f (b i) = f i := by
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Finite ι
    b : Basis ι R M
    f : Finsupp ι R
    i : ι
    ⊢ Eq (((Finsupp.linearCombination R b.coord) f) (b i)) (f i)
  -/
  haveI := Classical.decEq ι
  /-
    R : Type uR
    M : Type uM
    ι : Type uι
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Finite ι
    b : Basis ι R M
    f : Finsupp ι R
    i : ι
    this : DecidableEq ι
    ⊢ Eq (((Finsupp.linearCombination R b.coord) f) (b i)) (f i)
  -/
  rw [← coe_dualBasis, linearCombination_dualBasis]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias total_coord := linearCombination_coord


theorem dual_rank_eq [CommRing K] [AddCommGroup V] [Module K V] [Finite ι] (b : Basis ι K V) :
    Cardinal.lift.{uK,uV} (Module.rank K V) = Module.rank K (Dual K V) := by
  /-
    K : Type uK
    V : Type uV
    ι : Type uι
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Finite ι
    b : Basis ι K V
    ⊢ Eq (Cardinal.lift.{uK, uV} (Module.rank K V)) (Module.rank K (Module.Dual K  …
  -/
  classical rw [← lift_umax.{uV,uK}, b.toDualEquiv.lift_rank_eq, lift_id'.{uV,uK}]
  /-
    🎉 no goals
  -/


theorem eval_ker : LinearMap.ker (eval K V) = ⊥ :=
  have ⟨s, hs⟩ := Module.projective_def'.mp ‹Projective K V›
  ker_eq_bot.mpr <| .of_comp (f := s.dualMap.dualMap) <| (ker_eq_bot.mp <|
    Finsupp.basisSingleOne (R := K).eval_ker).comp (injective_of_comp_eq_id s _ hs)


theorem map_eval_injective : (Submodule.map (eval K V)).Injective := by
  /-
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    ⊢ Function.Injective (Submodule.map (Module.Dual.eval K V))
  -/
  apply Submodule.map_injective_of_injective
  /-
    case hf
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    ⊢ Function.Injective ⇑(Module.Dual.eval K V)
  -/
  rw [← LinearMap.ker_eq_bot]
  /-
    case hf
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    ⊢ Eq (LinearMap.ker (Module.Dual.eval K V)) Bot.bot
  -/
  exact eval_ker K V
  /-
    🎉 no goals
  -/


theorem comap_eval_surjective : (Submodule.comap (eval K V)).Surjective := by
  /-
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    ⊢ Function.Surjective (Submodule.comap (Module.Dual.eval K V))
  -/
  apply Submodule.comap_surjective_of_injective
  /-
    case hf
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    ⊢ Function.Injective ⇑(Module.Dual.eval K V)
  -/
  rw [← LinearMap.ker_eq_bot]
  /-
    case hf
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    ⊢ Eq (LinearMap.ker (Module.Dual.eval K V)) Bot.bot
  -/
  exact eval_ker K V
  /-
    🎉 no goals
  -/


theorem eval_apply_eq_zero_iff (v : V) : (eval K V) v = 0 ↔ v = 0 := by
  /-
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    v : V
    ⊢ Iff (Eq ((Module.Dual.eval K V) v) 0) (Eq v 0)
  -/
  simpa only using SetLike.ext_iff.mp (eval_ker K V) v
  /-
    🎉 no goals
  -/


theorem eval_apply_injective : Function.Injective (eval K V) :=
  (injective_iff_map_eq_zero' (eval K V)).mpr (eval_apply_eq_zero_iff K)


theorem forall_dual_apply_eq_zero_iff (v : V) : (∀ φ : Module.Dual K V, φ v = 0) ↔ v = 0 := by
  /-
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    v : V
    ⊢ Iff (∀ (φ : Module.Dual K V), Eq (φ v) 0) (Eq v 0)
  -/
  rw [← eval_apply_eq_zero_iff K v, LinearMap.ext_iff]
  /-
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    v : V
    ⊢ Iff (∀ (φ : Module.Dual K V), Eq (φ v) 0) (∀ (x : Module.Dual K V), Eq (((Mo …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem subsingleton_dual_iff :
    Subsingleton (Dual K V) ↔ Subsingleton V := by
  /-
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    ⊢ Iff (Subsingleton (Module.Dual K V)) (Subsingleton V)
  -/
  refine ⟨fun h ↦ ⟨fun v w ↦ ?_⟩, fun _ ↦ inferInstance⟩
  /-
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    h : Subsingleton (Module.Dual K V)
    v w : V
    ⊢ Eq v w
  -/
  rw [← sub_eq_zero, ← forall_dual_apply_eq_zero_iff K (v - w)]
  /-
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    h : Subsingleton (Module.Dual K V)
    v w : V
    ⊢ ∀ (φ : Module.Dual K V), Eq (φ (HSub.hSub v w)) 0
  -/
  intros f
  /-
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Projective K V
    h : Subsingleton (Module.Dual K V)
    v w : V
    f : Module.Dual K V
    ⊢ Eq (f (HSub.hSub v w)) 0
  -/
  simp [Subsingleton.elim f 0]
  /-
    🎉 no goals
  -/


@[simp]
theorem nontrivial_dual_iff :
    Nontrivial (Dual K V) ↔ Nontrivial V := by
  rw [← not_iff_not, not_nontrivial_iff_subsingleton, not_nontrivial_iff_subsingleton,
    subsingleton_dual_iff]


instance instNontrivialDual [Nontrivial V] : Nontrivial (Dual K V) :=
  (nontrivial_dual_iff K).mpr inferInstance


omit [Projective K V] in
theorem finite_dual_iff [Free K V] : Module.Finite K (Dual K V) ↔ Module.Finite K V := by
  /-
    K : Type uK
    V : Type uV
    inst✝³ : CommRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    ⊢ Iff (Module.Finite K (Module.Dual K V)) (Module.Finite K V)
  -/
  constructor <;> intro h
    /-
      case mp
      K : Type uK
      V : Type uV
      inst✝³ : CommRing K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      h : Module.Finite K (Module.Dual K V)
      ⊢ Module.Finite K V
    -/
  · obtain ⟨⟨ι, b⟩⟩ := Free.exists_basis (R := K) (M := V)
    /-
      case mp.intro.mk
      K : Type uK
      V : Type uV
      inst✝³ : CommRing K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      h : Module.Finite K (Module.Dual K V)
      ι : Type uV
      b : Basis ι K V
      ⊢ Module.Finite K V
    -/
    nontriviality K
    /-
      K : Type uK
      V : Type uV
      inst✝³ : CommRing K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      h : Module.Finite K (Module.Dual K V)
      ι : Type uV
      b : Basis ι K V
      a✝ : Nontrivial K
      ⊢ Module.Finite K V
    -/
    obtain ⟨⟨s, span_s⟩⟩ := h
    classical
    haveI := (b.linearIndependent.map' _ b.toDual_ker).finite_of_le_span_finite _ s ?_
    · exact Finite.of_basis b
    · rw [span_s]; apply le_top
    /-
      case mpr
      K : Type uK
      V : Type uV
      inst✝³ : CommRing K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      h : Module.Finite K V
      ⊢ Module.Finite K (Module.Dual K V)
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


theorem dual_rank_eq [Free K V] [Module.Finite K V] :
    Cardinal.lift.{uK,uV} (Module.rank K V) = Module.rank K (Dual K V) :=
  (Module.Free.chooseBasis K V).dual_rank_eq


/-- A reflexive module is one for which the natural map to its double dual is a bijection.

Any finitely-generated projective module (and thus any finite-dimensional vector space)
is reflexive. See `Module.instIsReflexiveOfFiniteOfProjective`. -/
class IsReflexive : Prop where
  /-- A reflexive module is one for which the natural map to its double dual is a bijection. -/
  bijective_dual_eval' : Bijective (Dual.eval R M)


lemma bijective_dual_eval [IsReflexive R M] : Bijective (Dual.eval R M) :=
  IsReflexive.bijective_dual_eval'


/-- See also `Module.instFiniteDimensionalOfIsReflexive` for the converse over a field. -/
instance (priority := 900) IsReflexive.of_finite_of_free [Module.Finite R M] [Free R M] :
    IsReflexive R M where
  bijective_dual_eval'.left := ker_eq_bot.mp (Free.chooseBasis R M).eval_ker
  bijective_dual_eval'.right := range_eq_top.mp (Free.chooseBasis R M).eval_range


theorem erange_coe : LinearMap.range (eval R M) = ⊤ :=
  range_eq_top.mpr (bijective_dual_eval _ _).2


/-- The bijection between a reflexive module and its double dual, bundled as a `LinearEquiv`. -/
def evalEquiv : M ≃ₗ[R] Dual R (Dual R M) :=
  LinearEquiv.ofBijective _ (bijective_dual_eval R M)


@[simp] lemma evalEquiv_toLinearMap : evalEquiv R M = Dual.eval R M := rfl


@[simp] lemma evalEquiv_apply (m : M) : evalEquiv R M m = Dual.eval R M m := rfl


@[simp] lemma apply_evalEquiv_symm_apply (f : Dual R M) (g : Dual R (Dual R M)) :
    f ((evalEquiv R M).symm g) = g f := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.IsReflexive R M
    f : Module.Dual R M
    g : Module.Dual R (Module.Dual R M)
    ⊢ Eq (f ((Module.evalEquiv R M).symm g)) (g f)
  -/
  set m := (evalEquiv R M).symm g
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.IsReflexive R M
    f : Module.Dual R M
    g : Module.Dual R (Module.Dual R M)
    m : M := (Module.evalEquiv R M).symm g
    ⊢ Eq (f m) (g f)
  -/
  rw [← (evalEquiv R M).apply_symm_apply g, evalEquiv_apply, Dual.eval_apply]
  /-
    🎉 no goals
  -/


@[simp] lemma symm_dualMap_evalEquiv :
    (evalEquiv R M).symm.dualMap = Dual.eval R (Dual R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.IsReflexive R M
    ⊢ Eq (↑(Module.evalEquiv R M).symm.dualMap) (Module.Dual.eval R (Module.Dual R …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp] lemma Dual.eval_comp_comp_evalEquiv_eq
    {M' : Type*} [AddCommGroup M'] [Module R M'] {f : M →ₗ[R] M'} :
    Dual.eval R M' ∘ₗ f ∘ₗ (evalEquiv R M).symm = f.dualMap.dualMap := by
  rw [← LinearMap.comp_assoc, LinearEquiv.comp_toLinearMap_symm_eq,
    evalEquiv_toLinearMap, eval_naturality]


lemma dualMap_dualMap_eq_iff_of_injective
    {M' : Type*} [AddCommGroup M'] [Module R M'] {f g : M →ₗ[R] M'}
    (h : Injective (Dual.eval R M')) :
    f.dualMap.dualMap = g.dualMap.dualMap ↔ f = g := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.IsReflexive R M
    M' : Type u_4
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f g : LinearMap (RingHom.id R) M M'
    h : Function.Injective ⇑(Module.Dual.eval R M')
    ⊢ Iff (Eq f.dualMap.dualMap g.dualMap.dualMap) (Eq f g)
  -/
  simp only [← Dual.eval_comp_comp_evalEquiv_eq]
  refine ⟨fun hfg => ?_, fun a ↦ congrArg (Dual.eval R M').comp
    (congrFun (congrArg LinearMap.comp a) (evalEquiv R M).symm.toLinearMap)⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.IsReflexive R M
    M' : Type u_4
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f g : LinearMap (RingHom.id R) M M'
    h : Function.Injective ⇑(Module.Dual.eval R M')
    hfg : Eq ((Module.Dual.eval R M').comp (f.comp ↑(Module.evalEquiv R M).symm))  …
    ⊢ Eq f g
  -/
  rw [propext (cancel_left h), LinearEquiv.eq_comp_toLinearMap_iff] at hfg
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.IsReflexive R M
    M' : Type u_4
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f g : LinearMap (RingHom.id R) M M'
    h : Function.Injective ⇑(Module.Dual.eval R M')
    hfg : Eq f g
    ⊢ Eq f g
  -/
  exact hfg
  /-
    🎉 no goals
  -/


@[simp] lemma dualMap_dualMap_eq_iff
    {M' : Type*} [AddCommGroup M'] [Module R M'] [IsReflexive R M'] {f g : M →ₗ[R] M'} :
    f.dualMap.dualMap = g.dualMap.dualMap ↔ f = g :=
  dualMap_dualMap_eq_iff_of_injective _ _ (bijective_dual_eval R M').injective


/-- The dual of a reflexive module is reflexive. -/
instance Dual.instIsReflecive : IsReflexive R (Dual R M) :=
      /-
        K : Type uK
        V : Type uV
        inst✝⁹ : CommRing K
        inst✝⁸ : AddCommGroup V
        inst✝⁷ : Module K V
        inst✝⁶ : Module.Projective K V
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : AddCommGroup M
        inst✝³ : AddCommGroup N
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module.IsReflexive R M
        ⊢ Function.Bijective ⇑(Module.Dual.eval R (Module.Dual R M))
      -/
  ⟨by simpa only [← symm_dualMap_evalEquiv] using (evalEquiv R M).dualMap.symm.bijective⟩
      /-
        🎉 no goals
      -/


variable {R M N} in
/-- A direct summand of a reflexive module is reflexive. -/
lemma IsReflexive.of_split (i : N →ₗ[R] M) (s : M →ₗ[R] N) (H : s ∘ₗ i = .id) :
    IsReflexive R N where
  bijective_dual_eval' :=
    ⟨.of_comp (f := i.dualMap.dualMap) <|
      (bijective_dual_eval R M).1.comp (injective_of_comp_eq_id i _ H),
    .of_comp (g := s) <| (surjective_of_comp_eq_id i.dualMap.dualMap s.dualMap.dualMap <|
      congr_arg (dualMap ∘ dualMap) H).comp (bijective_dual_eval R M).2⟩


instance (priority := 900) [Module.Finite R N] [Projective R N] : IsReflexive R N :=
  have ⟨_, f, hf⟩ := Finite.exists_fin' R N
  have ⟨g, H⟩ := projective_lifting_property f .id hf
  .of_split g f H


/-- The isomorphism `Module.evalEquiv` induces an order isomorphism on subspaces. -/
def mapEvalEquiv : Submodule R M ≃o Submodule R (Dual R (Dual R M)) :=
  Submodule.orderIsoMapComap (evalEquiv R M)


@[simp]
theorem mapEvalEquiv_apply (W : Submodule R M) :
    mapEvalEquiv R M W = W.map (Dual.eval R M) :=
  rfl


@[simp]
theorem mapEvalEquiv_symm_apply (W'' : Submodule R (Dual R (Dual R M))) :
    (mapEvalEquiv R M).symm W'' = W''.comap (Dual.eval R M) :=
  rfl


instance _root_.Prod.instModuleIsReflexive [IsReflexive R N] :
    IsReflexive R (M × N) where
  bijective_dual_eval' := by
    let e : Dual R (Dual R (M × N)) ≃ₗ[R] Dual R (Dual R M) × Dual R (Dual R N) :=
      (dualProdDualEquivDual R M N).dualMap.trans
        (dualProdDualEquivDual R (Dual R M) (Dual R N)).symm
    have : Dual.eval R (M × N) = e.symm.comp ((Dual.eval R M).prodMap (Dual.eval R N)) := by
      ext m f <;> simp [e]
    simp only [this, LinearEquiv.trans_symm, LinearEquiv.symm_symm, LinearEquiv.dualMap_symm,
      coe_comp, LinearEquiv.coe_coe, EquivLike.comp_bijective]
    /-
      K : Type uK
      V : Type uV
      inst✝¹⁰ : CommRing K
      inst✝⁹ : AddCommGroup V
      inst✝⁸ : Module K V
      inst✝⁷ : Module.Projective K V
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module.IsReflexive R M
      inst✝ : Module.IsReflexive R N
      e : LinearEquiv (RingHom.id R) (Module.Dual R (Module.Dual R (Prod M N))) (Pro …
      this : Eq (Module.Dual.eval R (Prod M N)) ((↑e.symm).comp ((Module.Dual.eval R …
      ⊢ Function.Bijective ⇑((Module.Dual.eval R M).prodMap (Module.Dual.eval R N))
    -/
    exact (bijective_dual_eval R M).prodMap (bijective_dual_eval R N)
    /-
      🎉 no goals
    -/


variable {R M N} in
lemma equiv (e : M ≃ₗ[R] N) : IsReflexive R N where
  bijective_dual_eval' := by
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module.IsReflexive R M
      e : LinearEquiv (RingHom.id R) M N
      ⊢ Function.Bijective ⇑(Module.Dual.eval R N)
    -/
    let ed : Dual R (Dual R N) ≃ₗ[R] Dual R (Dual R M) := e.symm.dualMap.dualMap
    have : Dual.eval R N = ed.symm.comp ((Dual.eval R M).comp e.symm.toLinearMap) := by
      ext m f
      exact DFunLike.congr_arg f (e.apply_symm_apply m).symm
    simp only [this, LinearEquiv.trans_symm, LinearEquiv.symm_symm, LinearEquiv.dualMap_symm,
      coe_comp, LinearEquiv.coe_coe, EquivLike.comp_bijective]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module.IsReflexive R M
      e : LinearEquiv (RingHom.id R) M N
      ed : LinearEquiv (RingHom.id R) (Module.Dual R (Module.Dual R N)) (Module.Dual …
      this : Eq (Module.Dual.eval R N) ((↑ed.symm).comp ((Module.Dual.eval R M).comp …
      ⊢ Function.Bijective (Function.comp ⇑(Module.Dual.eval R M) ⇑e.symm)
    -/
    exact Bijective.comp (bijective_dual_eval R M) (LinearEquiv.bijective _)
    /-
      🎉 no goals
    -/


instance _root_.MulOpposite.instModuleIsReflexive : IsReflexive R (MulOpposite M) :=
  equiv <| MulOpposite.opLinearEquiv _


instance _root_.ULift.instModuleIsReflexive.{w} : IsReflexive R (ULift.{w} M) :=
  equiv ULift.moduleEquiv.symm


instance instFiniteDimensionalOfIsReflexive (K V : Type*)
    [Field K] [AddCommGroup V] [Module K V] [IsReflexive K V] :
    FiniteDimensional K V := by
  /-
    K✝ : Type uK
    V✝ : Type uV
    inst✝¹³ : CommRing K✝
    inst✝¹² : AddCommGroup V✝
    inst✝¹¹ : Module K✝ V✝
    inst✝¹⁰ : Module.Projective K✝ V✝
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Module.IsReflexive R M
    K : Type u_4
    V : Type u_5
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.IsReflexive K V
    ⊢ FiniteDimensional K V
  -/
  rw [FiniteDimensional, ← rank_lt_aleph0_iff]
  /-
    K✝ : Type uK
    V✝ : Type uV
    inst✝¹³ : CommRing K✝
    inst✝¹² : AddCommGroup V✝
    inst✝¹¹ : Module K✝ V✝
    inst✝¹⁰ : Module.Projective K✝ V✝
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Module.IsReflexive R M
    K : Type u_4
    V : Type u_5
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.IsReflexive K V
    ⊢ LT.lt (Module.rank K V) Cardinal.aleph0
  -/
  by_contra! contra
  suffices lift (Module.rank K V) < Module.rank K (Dual K (Dual K V)) by
    have heq := lift_rank_eq_of_equiv_equiv (R := K) (R' := K) (M := V) (M' := Dual K (Dual K V))
      (ZeroHom.id K) (evalEquiv K V) bijective_id (fun r v ↦ (evalEquiv K V).map_smul _ _)
    rw [← lift_umax, heq, lift_id'] at this
    exact lt_irrefl _ this
  /-
    K✝ : Type uK
    V✝ : Type uV
    inst✝¹³ : CommRing K✝
    inst✝¹² : AddCommGroup V✝
    inst✝¹¹ : Module K✝ V✝
    inst✝¹⁰ : Module.Projective K✝ V✝
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Module.IsReflexive R M
    K : Type u_4
    V : Type u_5
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.IsReflexive K V
    contra : LE.le Cardinal.aleph0 (Module.rank K V)
    ⊢ LT.lt (Cardinal.lift.{u_4, u_5} (Module.rank K V)) (Module.rank K (Module.Du …
  -/
  have h₁ : lift (Module.rank K V) < Module.rank K (Dual K V) := lift_rank_lt_rank_dual contra
  have h₂ : Module.rank K (Dual K V) < Module.rank K (Dual K (Dual K V)) := by
    convert lift_rank_lt_rank_dual <| le_trans (by simpa) h₁.le
    rw [lift_id']
  /-
    K✝ : Type uK
    V✝ : Type uV
    inst✝¹³ : CommRing K✝
    inst✝¹² : AddCommGroup V✝
    inst✝¹¹ : Module K✝ V✝
    inst✝¹⁰ : Module.Projective K✝ V✝
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Module.IsReflexive R M
    K : Type u_4
    V : Type u_5
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.IsReflexive K V
    contra : LE.le Cardinal.aleph0 (Module.rank K V)
    h₁ : LT.lt (Cardinal.lift.{u_4, u_5} (Module.rank K V)) (Module.rank K (Module …
    h₂ : LT.lt (Module.rank K (Module.Dual K V)) (Module.rank K (Module.Dual K (Mo …
    ⊢ LT.lt (Cardinal.lift.{u_4, u_5} (Module.rank K V)) (Module.rank K (Module.Du …
  -/
  exact lt_trans h₁ h₂
  /-
    🎉 no goals
  -/


instance [IsDomain R] : NoZeroSMulDivisors R M := by
  /-
    K : Type uK
    V : Type uV
    inst✝¹⁰ : CommRing K
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module K V
    inst✝⁷ : Module.Projective K V
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module.IsReflexive R M
    inst✝ : IsDomain R
    ⊢ NoZeroSMulDivisors R M
  -/
  refine (noZeroSMulDivisors_iff R M).mpr ?_
  /-
    K : Type uK
    V : Type uV
    inst✝¹⁰ : CommRing K
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module K V
    inst✝⁷ : Module.Projective K V
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module.IsReflexive R M
    inst✝ : IsDomain R
    ⊢ ∀ {c : R} {x : M}, Eq (HSMul.hSMul c x) 0 → Or (Eq c 0) (Eq x 0)
  -/
  intro r m hrm
  /-
    K : Type uK
    V : Type uV
    inst✝¹⁰ : CommRing K
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module K V
    inst✝⁷ : Module.Projective K V
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module.IsReflexive R M
    inst✝ : IsDomain R
    r : R
    m : M
    hrm : Eq (HSMul.hSMul r m) 0
    ⊢ Or (Eq r 0) (Eq m 0)
  -/
  rw [or_iff_not_imp_left]
  /-
    K : Type uK
    V : Type uV
    inst✝¹⁰ : CommRing K
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module K V
    inst✝⁷ : Module.Projective K V
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module.IsReflexive R M
    inst✝ : IsDomain R
    r : R
    m : M
    hrm : Eq (HSMul.hSMul r m) 0
    ⊢ Not (Eq r 0) → Eq m 0
  -/
  intro hr
  /-
    K : Type uK
    V : Type uV
    inst✝¹⁰ : CommRing K
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module K V
    inst✝⁷ : Module.Projective K V
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module.IsReflexive R M
    inst✝ : IsDomain R
    r : R
    m : M
    hrm : Eq (HSMul.hSMul r m) 0
    hr : Not (Eq r 0)
    ⊢ Eq m 0
  -/
  suffices Dual.eval R M m = Dual.eval R M 0 from (bijective_dual_eval R M).injective this
  /-
    K : Type uK
    V : Type uV
    inst✝¹⁰ : CommRing K
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module K V
    inst✝⁷ : Module.Projective K V
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module.IsReflexive R M
    inst✝ : IsDomain R
    r : R
    m : M
    hrm : Eq (HSMul.hSMul r m) 0
    hr : Not (Eq r 0)
    ⊢ Eq ((Module.Dual.eval R M) m) ((Module.Dual.eval R M) 0)
  -/
  ext n
  /-
    case h
    K : Type uK
    V : Type uV
    inst✝¹⁰ : CommRing K
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module K V
    inst✝⁷ : Module.Projective K V
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module.IsReflexive R M
    inst✝ : IsDomain R
    r : R
    m : M
    hrm : Eq (HSMul.hSMul r m) 0
    hr : Not (Eq r 0)
    n : Module.Dual R M
    ⊢ Eq (((Module.Dual.eval R M) m) n) (((Module.Dual.eval R M) 0) n)
  -/
  simp only [Dual.eval_apply, map_zero, LinearMap.zero_apply]
  /-
    case h
    K : Type uK
    V : Type uV
    inst✝¹⁰ : CommRing K
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module K V
    inst✝⁷ : Module.Projective K V
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module.IsReflexive R M
    inst✝ : IsDomain R
    r : R
    m : M
    hrm : Eq (HSMul.hSMul r m) 0
    hr : Not (Eq r 0)
    n : Module.Dual R M
    ⊢ Eq (n m) 0
  -/
  suffices r • n m = 0 from eq_zero_of_ne_zero_of_mul_left_eq_zero hr this
  /-
    case h
    K : Type uK
    V : Type uV
    inst✝¹⁰ : CommRing K
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module K V
    inst✝⁷ : Module.Projective K V
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module.IsReflexive R M
    inst✝ : IsDomain R
    r : R
    m : M
    hrm : Eq (HSMul.hSMul r m) 0
    hr : Not (Eq r 0)
    n : Module.Dual R M
    ⊢ Eq (HSMul.hSMul r (n m)) 0
  -/
  rw [← LinearMap.map_smul_of_tower, hrm, LinearMap.map_zero]
  /-
    🎉 no goals
  -/


theorem exists_dual_map_eq_bot_of_nmem {x : M} (hx : x ∉ p) (hp' : Free R (M ⧸ p)) :
    ∃ f : Dual R M, f x ≠ 0 ∧ p.map f = ⊥ := by
  suffices ∃ f : Dual R (M ⧸ p), f (p.mkQ x) ≠ 0 by
    obtain ⟨f, hf⟩ := this; exact ⟨f.comp p.mkQ, hf, by simp [Submodule.map_comp]⟩
  rwa [← Submodule.Quotient.mk_eq_zero, ← Submodule.mkQ_apply,
    ← forall_dual_apply_eq_zero_iff (K := R), not_forall] at hx


theorem exists_dual_map_eq_bot_of_lt_top (hp : p < ⊤) (hp' : Free R (M ⧸ p)) :
    ∃ f : Dual R M, f ≠ 0 ∧ p.map f = ⊥ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    hp : LT.lt p Top.top
    hp' : Module.Free R (HasQuotient.Quotient M p)
    ⊢ Exists fun f => And (Ne f 0) (Eq (Submodule.map f p) Bot.bot)
  -/
  obtain ⟨x, hx⟩ : ∃ x : M, x ∉ p := by rw [lt_top_iff_ne_top] at hp; contrapose! hp; ext; simp [hp]
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    hp : LT.lt p Top.top
    hp' : Module.Free R (HasQuotient.Quotient M p)
    x : M
    hx : Not (Membership.mem p x)
    ⊢ Exists fun f => And (Ne f 0) (Eq (Submodule.map f p) Bot.bot)
  -/
  obtain ⟨f, hf, hf'⟩ := p.exists_dual_map_eq_bot_of_nmem hx hp'
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    hp : LT.lt p Top.top
    hp' : Module.Free R (HasQuotient.Quotient M p)
    x : M
    hx : Not (Membership.mem p x)
    f : Module.Dual R M
    hf : Ne (f x) 0
    hf' : Eq (Submodule.map f p) Bot.bot
    ⊢ Exists fun f => And (Ne f 0) (Eq (Submodule.map f p) Bot.bot)
  -/
  exact ⟨f, by aesop, hf'⟩
  /-
    🎉 no goals
  -/


/-- Consider a reflexive module and a set `s` of linear forms. If for any `z ≠ 0` there exists
`f ∈ s` such that `f z ≠ 0`, then `s` spans the whole dual space. -/
theorem span_eq_top_of_ne_zero [IsReflexive R M]
    {s : Set (M →ₗ[R] R)} [Free R ((M →ₗ[R] R) ⧸ (span R s))]
    (h : ∀ z ≠ 0, ∃ f ∈ s, f z ≠ 0) : span R s = ⊤ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.IsReflexive R M
    s : Set (LinearMap (RingHom.id R) M R)
    inst✝ : Module.Free R (HasQuotient.Quotient (LinearMap (RingHom.id R) M R) (Su …
    h : ∀ (z : M), Ne z 0 → Exists fun f => And (Membership.mem s f) (Ne (f z) 0)
    ⊢ Eq (Submodule.span R s) Top.top
  -/
  by_contra! hn
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.IsReflexive R M
    s : Set (LinearMap (RingHom.id R) M R)
    inst✝ : Module.Free R (HasQuotient.Quotient (LinearMap (RingHom.id R) M R) (Su …
    h : ∀ (z : M), Ne z 0 → Exists fun f => And (Membership.mem s f) (Ne (f z) 0)
    hn : Ne (Submodule.span R s) Top.top
    ⊢ False
  -/
  obtain ⟨φ, φne, hφ⟩ := exists_dual_map_eq_bot_of_lt_top hn.lt_top inferInstance
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.IsReflexive R M
    s : Set (LinearMap (RingHom.id R) M R)
    inst✝ : Module.Free R (HasQuotient.Quotient (LinearMap (RingHom.id R) M R) (Su …
    h : ∀ (z : M), Ne z 0 → Exists fun f => And (Membership.mem s f) (Ne (f z) 0)
    hn : Ne (Submodule.span R s) Top.top
    φ : Module.Dual R (LinearMap (RingHom.id R) M R)
    φne : Ne φ 0
    hφ : Eq (Submodule.map φ (Submodule.span R s)) Bot.bot
    ⊢ False
  -/
  let φs := (evalEquiv R M).symm φ
  have this f (hf : f ∈ s) : f φs = 0 := by
    rw [← mem_bot R, ← hφ, mem_map]
    exact ⟨f, subset_span hf, (apply_evalEquiv_symm_apply R M f φ).symm⟩
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.IsReflexive R M
    s : Set (LinearMap (RingHom.id R) M R)
    inst✝ : Module.Free R (HasQuotient.Quotient (LinearMap (RingHom.id R) M R) (Su …
    h : ∀ (z : M), Ne z 0 → Exists fun f => And (Membership.mem s f) (Ne (f z) 0)
    hn : Ne (Submodule.span R s) Top.top
    φ : Module.Dual R (LinearMap (RingHom.id R) M R)
    φne : Ne φ 0
    hφ : Eq (Submodule.map φ (Submodule.span R s)) Bot.bot
    φs : M := (Module.evalEquiv R M).symm φ
    this : ∀ (f : LinearMap (RingHom.id R) M R), Membership.mem s f → Eq (f φs) 0
    ⊢ False
  -/
  obtain ⟨x, xs, hx⟩ := h φs (by simp [φne, φs])
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.IsReflexive R M
    s : Set (LinearMap (RingHom.id R) M R)
    inst✝ : Module.Free R (HasQuotient.Quotient (LinearMap (RingHom.id R) M R) (Su …
    h : ∀ (z : M), Ne z 0 → Exists fun f => And (Membership.mem s f) (Ne (f z) 0)
    hn : Ne (Submodule.span R s) Top.top
    φ : Module.Dual R (LinearMap (RingHom.id R) M R)
    φne : Ne φ 0
    hφ : Eq (Submodule.map φ (Submodule.span R s)) Bot.bot
    φs : M := (Module.evalEquiv R M).symm φ
    this : ∀ (f : LinearMap (RingHom.id R) M R), Membership.mem s f → Eq (f φs) 0
    x : LinearMap (RingHom.id R) M R
    xs : Membership.mem s x
    hx : Ne (x φs) 0
    ⊢ False
  -/
  exact hx <| this x xs
  /-
    🎉 no goals
  -/


theorem _root_.FiniteDimensional.mem_span_of_iInf_ker_le_ker [FiniteDimensional 𝕜 E]
    {L : ι → E →ₗ[𝕜] 𝕜} {K : E →ₗ[𝕜] 𝕜}
    (h : ⨅ i, LinearMap.ker (L i) ≤ ker K) : K ∈ span 𝕜 (range L) := by
  /-
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.range L)) K
  -/
  by_contra hK
  /-
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    hK : Not (Membership.mem (Submodule.span 𝕜 (Set.range L)) K)
    ⊢ False
  -/
  rcases exists_dual_map_eq_bot_of_nmem hK inferInstance with ⟨φ, φne, hφ⟩
  /-
    case intro.intro
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    hK : Not (Membership.mem (Submodule.span 𝕜 (Set.range L)) K)
    φ : Module.Dual 𝕜 (LinearMap (RingHom.id 𝕜) E 𝕜)
    φne : Ne (φ K) 0
    hφ : Eq (Submodule.map φ (Submodule.span 𝕜 (Set.range L))) Bot.bot
    ⊢ False
  -/
  let φs := (Module.evalEquiv 𝕜 E).symm φ
  have : K φs = 0 := by
    refine h <| (Submodule.mem_iInf _).2 fun i ↦ (mem_bot 𝕜).1 ?_
    rw [← hφ, Submodule.mem_map]
    exact ⟨L i, Submodule.subset_span ⟨i, rfl⟩, (apply_evalEquiv_symm_apply 𝕜 E _ φ).symm⟩
  /-
    case intro.intro
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    hK : Not (Membership.mem (Submodule.span 𝕜 (Set.range L)) K)
    φ : Module.Dual 𝕜 (LinearMap (RingHom.id 𝕜) E 𝕜)
    φne : Ne (φ K) 0
    hφ : Eq (Submodule.map φ (Submodule.span 𝕜 (Set.range L))) Bot.bot
    φs : E := (Module.evalEquiv 𝕜 E).symm φ
    this : Eq (K φs) 0
    ⊢ False
  -/
  simp only [apply_evalEquiv_symm_apply, φs, φne] at this
  /-
    🎉 no goals
  -/


/-- Given some linear forms $L_1, ..., L_n, K$ over a vector space $E$, if
$\bigcap_{i=1}^n \mathrm{ker}(L_i) \subseteq \mathrm{ker}(K)$, then $K$ is in the space generated
by $L_1, ..., L_n$. -/
theorem _root_.mem_span_of_iInf_ker_le_ker [Finite ι] {L : ι → E →ₗ[𝕜] 𝕜} {K : E →ₗ[𝕜] 𝕜}
    (h : ⨅ i, ker (L i) ≤ ker K) : K ∈ span 𝕜 (range L) := by
  /-
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.range L)) K
  -/
  have _ := Fintype.ofFinite ι
  /-
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    x✝ : Fintype ι
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.range L)) K
  -/
  let φ : E →ₗ[𝕜] ι → 𝕜 := LinearMap.pi L
  /-
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    x✝ : Fintype ι
    φ : LinearMap (RingHom.id 𝕜) E (ι → 𝕜) := LinearMap.pi L
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.range L)) K
  -/
  let p := ⨅ i, ker (L i)
  /-
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    x✝ : Fintype ι
    φ : LinearMap (RingHom.id 𝕜) E (ι → 𝕜) := LinearMap.pi L
    p : Submodule 𝕜 E := iInf fun i => LinearMap.ker (L i)
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.range L)) K
  -/
  have p_eq : p = ker φ := (ker_pi L).symm
  /-
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    x✝ : Fintype ι
    φ : LinearMap (RingHom.id 𝕜) E (ι → 𝕜) := LinearMap.pi L
    p : Submodule 𝕜 E := iInf fun i => LinearMap.ker (L i)
    p_eq : Eq p (LinearMap.ker φ)
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.range L)) K
  -/
  let ψ : (E ⧸ p) →ₗ[𝕜] ι → 𝕜 := p.liftQ φ p_eq.le
  /-
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    x✝ : Fintype ι
    φ : LinearMap (RingHom.id 𝕜) E (ι → 𝕜) := LinearMap.pi L
    p : Submodule 𝕜 E := iInf fun i => LinearMap.ker (L i)
    p_eq : Eq p (LinearMap.ker φ)
    ψ : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) (ι → 𝕜) := p.liftQ φ ⋯
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.range L)) K
  -/
  have _ : FiniteDimensional 𝕜 (E ⧸ p) := of_injective ψ (ker_eq_bot.1 (ker_liftQ_eq_bot' p φ p_eq))
  /-
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    x✝¹ : Fintype ι
    φ : LinearMap (RingHom.id 𝕜) E (ι → 𝕜) := LinearMap.pi L
    p : Submodule 𝕜 E := iInf fun i => LinearMap.ker (L i)
    p_eq : Eq p (LinearMap.ker φ)
    ψ : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) (ι → 𝕜) := p.liftQ φ ⋯
    x✝ : FiniteDimensional 𝕜 (HasQuotient.Quotient E p)
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.range L)) K
  -/
  let L' i : (E ⧸ p) →ₗ[𝕜] 𝕜 := p.liftQ (L i) (iInf_le _ i)
  /-
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    x✝¹ : Fintype ι
    φ : LinearMap (RingHom.id 𝕜) E (ι → 𝕜) := LinearMap.pi L
    p : Submodule 𝕜 E := iInf fun i => LinearMap.ker (L i)
    p_eq : Eq p (LinearMap.ker φ)
    ψ : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) (ι → 𝕜) := p.liftQ φ ⋯
    x✝ : FiniteDimensional 𝕜 (HasQuotient.Quotient E p)
    L' : ι → LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) 𝕜 := fun i => p.l …
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.range L)) K
  -/
  let K' : (E ⧸ p) →ₗ[𝕜] 𝕜 := p.liftQ K h
  have : ⨅ i, ker (L' i) ≤ ker K' := by
    simp_rw [← ker_pi, L', pi_liftQ_eq_liftQ_pi, ker_liftQ_eq_bot' p φ p_eq]
    exact bot_le
  obtain ⟨c, hK'⟩ :=
    (mem_span_range_iff_exists_fun 𝕜).1 (FiniteDimensional.mem_span_of_iInf_ker_le_ker this)
  /-
    case intro
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    x✝¹ : Fintype ι
    φ : LinearMap (RingHom.id 𝕜) E (ι → 𝕜) := LinearMap.pi L
    p : Submodule 𝕜 E := iInf fun i => LinearMap.ker (L i)
    p_eq : Eq p (LinearMap.ker φ)
    ψ : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) (ι → 𝕜) := p.liftQ φ ⋯
    x✝ : FiniteDimensional 𝕜 (HasQuotient.Quotient E p)
    L' : ι → LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) 𝕜 := fun i => p.l …
    K' : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) 𝕜 := p.liftQ K h
    this : LE.le (iInf fun i => LinearMap.ker (L' i)) (LinearMap.ker K')
    c : ι → 𝕜
    hK' : Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (L' i)) K'
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.range L)) K
  -/
  refine (mem_span_range_iff_exists_fun 𝕜).2 ⟨c, ?_⟩
  /-
    case intro
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    x✝¹ : Fintype ι
    φ : LinearMap (RingHom.id 𝕜) E (ι → 𝕜) := LinearMap.pi L
    p : Submodule 𝕜 E := iInf fun i => LinearMap.ker (L i)
    p_eq : Eq p (LinearMap.ker φ)
    ψ : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) (ι → 𝕜) := p.liftQ φ ⋯
    x✝ : FiniteDimensional 𝕜 (HasQuotient.Quotient E p)
    L' : ι → LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) 𝕜 := fun i => p.l …
    K' : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) 𝕜 := p.liftQ K h
    this : LE.le (iInf fun i => LinearMap.ker (L' i)) (LinearMap.ker K')
    c : ι → 𝕜
    hK' : Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (L' i)) K'
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (L i)) K
  -/
  conv_lhs => enter [2]; intro i; rw [← p.liftQ_mkQ (L i) (iInf_le _ i)]
  /-
    case intro
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    x✝¹ : Fintype ι
    φ : LinearMap (RingHom.id 𝕜) E (ι → 𝕜) := LinearMap.pi L
    p : Submodule 𝕜 E := iInf fun i => LinearMap.ker (L i)
    p_eq : Eq p (LinearMap.ker φ)
    ψ : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) (ι → 𝕜) := p.liftQ φ ⋯
    x✝ : FiniteDimensional 𝕜 (HasQuotient.Quotient E p)
    L' : ι → LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) 𝕜 := fun i => p.l …
    K' : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) 𝕜 := p.liftQ K h
    this : LE.le (iInf fun i => LinearMap.ker (L' i)) (LinearMap.ker K')
    c : ι → 𝕜
    hK' : Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (L' i)) K'
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) ((p.liftQ (L i) ⋯).comp p.mkQ …
  -/
  rw [← p.liftQ_mkQ K h]
  /-
    case intro
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    x✝¹ : Fintype ι
    φ : LinearMap (RingHom.id 𝕜) E (ι → 𝕜) := LinearMap.pi L
    p : Submodule 𝕜 E := iInf fun i => LinearMap.ker (L i)
    p_eq : Eq p (LinearMap.ker φ)
    ψ : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) (ι → 𝕜) := p.liftQ φ ⋯
    x✝ : FiniteDimensional 𝕜 (HasQuotient.Quotient E p)
    L' : ι → LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) 𝕜 := fun i => p.l …
    K' : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) 𝕜 := p.liftQ K h
    this : LE.le (iInf fun i => LinearMap.ker (L' i)) (LinearMap.ker K')
    c : ι → 𝕜
    hK' : Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (L' i)) K'
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) ((p.liftQ (L i) ⋯).comp p.mkQ …
  -/
  ext x
  /-
    case intro.h
    ι : Type u_3
    𝕜 : Type u_4
    E : Type u_5
    inst✝³ : Field 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Finite ι
    L : ι → LinearMap (RingHom.id 𝕜) E 𝕜
    K : LinearMap (RingHom.id 𝕜) E 𝕜
    h : LE.le (iInf fun i => LinearMap.ker (L i)) (LinearMap.ker K)
    x✝¹ : Fintype ι
    φ : LinearMap (RingHom.id 𝕜) E (ι → 𝕜) := LinearMap.pi L
    p : Submodule 𝕜 E := iInf fun i => LinearMap.ker (L i)
    p_eq : Eq p (LinearMap.ker φ)
    ψ : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) (ι → 𝕜) := p.liftQ φ ⋯
    x✝ : FiniteDimensional 𝕜 (HasQuotient.Quotient E p)
    L' : ι → LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) 𝕜 := fun i => p.l …
    K' : LinearMap (RingHom.id 𝕜) (HasQuotient.Quotient E p) 𝕜 := p.liftQ K h
    this : LE.le (iInf fun i => LinearMap.ker (L' i)) (LinearMap.ker K')
    c : ι → 𝕜
    hK' : Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (L' i)) K'
    x : E
    ⊢ Eq ((Finset.univ.sum fun i => HSMul.hSMul (c i) ((p.liftQ (L i) ⋯).comp p.mk …
  -/
  convert LinearMap.congr_fun hK' (p.mkQ x)
  simp only [L',coeFn_sum, Finset.sum_apply, smul_apply, coe_comp, Function.comp_apply,
    smul_eq_mul]


open Lean.Elab.Tactic in
/-- Try using `Set.toFinite` to dispatch a `Set.Finite` goal. -/
def evalUseFiniteInstance : TacticM Unit := do
  evalTactic (← `(tactic| intros; apply Set.toFinite))


elab "use_finite_instance" : tactic => evalUseFiniteInstance


/-- `e` and `ε` have characteristic properties of a basis and its dual -/
-- @[nolint has_nonempty_instance] Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed
structure Module.DualBases (e : ι → M) (ε : ι → Dual R M) : Prop where
  eval_same : ∀ i, ε i (e i) = 1
  eval_of_ne : Pairwise fun i j ↦ ε i (e j) = 0
  protected total : ∀ {m : M}, (∀ i, ε i m = 0) → m = 0
  protected finite : ∀ m : M, {i | ε i m ≠ 0}.Finite := by use_finite_instance


/-- The coefficients of `v` on the basis `e` -/
def coeffs (h : DualBases e ε) (m : M) : ι →₀ R where
  toFun i := ε i m
  support := (h.finite m).toFinset
                            /-
                              R : Type u_1
                              M : Type u_2
                              ι : Type u_3
                              inst✝² : CommRing R
                              inst✝¹ : AddCommGroup M
                              inst✝ : Module R M
                              e : ι → M
                              ε : ι → Module.Dual R M
                              h : Module.DualBases e ε
                              m : M
                              i : ι
                              ⊢ Iff (Membership.mem ⋯.toFinset i) (Ne ((fun i => (ε i) m) i) 0)
                            -/
  mem_support_toFun i := by rw [Set.Finite.mem_toFinset, Set.mem_setOf_eq]
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem coeffs_apply (h : DualBases e ε) (m : M) (i : ι) : h.coeffs m i = ε i m :=
  rfl


/-- linear combinations of elements of `e`.
This is a convenient abbreviation for `Finsupp.linearCombination R e l` -/
def lc {ι} (e : ι → M) (l : ι →₀ R) : M :=
  l.sum fun (i : ι) (a : R) => a • e i


theorem lc_def (e : ι → M) (l : ι →₀ R) : lc e l = Finsupp.linearCombination R e l :=
  rfl


theorem dual_lc (l : ι →₀ R) (i : ι) : ε i (DualBases.lc e l) = l i := by
  /-
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    l : Finsupp ι R
    i : ι
    ⊢ Eq ((ε i) (Module.DualBases.lc e l)) (l i)
  -/
  rw [lc, _root_.map_finsupp_sum, Finsupp.sum_eq_single i (g := fun a b ↦ (ε i) (b • e a))]
  -- Porting note: cannot get at •
  -- simp only [h.eval, map_smul, smul_eq_mul]
    /-
      R : Type u_1
      M : Type u_2
      ι : Type u_3
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      e : ι → M
      ε : ι → Module.Dual R M
      h : Module.DualBases e ε
      l : Finsupp ι R
      i : ι
      ⊢ Eq ((ε i) (HSMul.hSMul (l i) (e i))) (l i)
    -/
  · simp [h.eval_same, smul_eq_mul]
    /-
      🎉 no goals
    -/
    /-
      case h₀
      R : Type u_1
      M : Type u_2
      ι : Type u_3
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      e : ι → M
      ε : ι → Module.Dual R M
      h : Module.DualBases e ε
      l : Finsupp ι R
      i : ι
      ⊢ ∀ (b : ι), Ne (l b) 0 → Ne b i → Eq ((ε i) (HSMul.hSMul (l b) (e b))) 0
    -/
  · intro q _ q_ne
    /-
      case h₀
      R : Type u_1
      M : Type u_2
      ι : Type u_3
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      e : ι → M
      ε : ι → Module.Dual R M
      h : Module.DualBases e ε
      l : Finsupp ι R
      i q : ι
      a✝ : Ne (l q) 0
      q_ne : Ne q i
      ⊢ Eq ((ε i) (HSMul.hSMul (l q) (e q))) 0
    -/
    simp [h.eval_of_ne q_ne.symm, smul_eq_mul]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      R : Type u_1
      M : Type u_2
      ι : Type u_3
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      e : ι → M
      ε : ι → Module.Dual R M
      h : Module.DualBases e ε
      l : Finsupp ι R
      i : ι
      ⊢ Eq (l i) 0 → Eq ((ε i) (HSMul.hSMul 0 (e i))) 0
    -/
  · simp
    /-
      🎉 no goals
    -/


@[simp]
theorem coeffs_lc (l : ι →₀ R) : h.coeffs (DualBases.lc e l) = l := by
  /-
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    l : Finsupp ι R
    ⊢ Eq (h.coeffs (Module.DualBases.lc e l)) l
  -/
  ext i
  /-
    case h
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    l : Finsupp ι R
    i : ι
    ⊢ Eq ((h.coeffs (Module.DualBases.lc e l)) i) (l i)
  -/
  rw [h.coeffs_apply, h.dual_lc]
  /-
    🎉 no goals
  -/


/-- For any m : M n, \sum_{p ∈ Q n} (ε p m) • e p = m -/
@[simp]
theorem lc_coeffs (m : M) : DualBases.lc e (h.coeffs m) = m := by
  /-
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    m : M
    ⊢ Eq (Module.DualBases.lc e (h.coeffs m)) m
  -/
  refine eq_of_sub_eq_zero <| h.total fun i ↦ ?_
  /-
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    m : M
    i : ι
    ⊢ Eq ((ε i) (HSub.hSub (Module.DualBases.lc e (h.coeffs m)) m)) 0
  -/
  simp [LinearMap.map_sub, h.dual_lc, sub_eq_zero]
  /-
    🎉 no goals
  -/


/-- `(h : DualBases e ε).basis` shows the family of vectors `e` forms a basis. -/
@[simps repr_apply, simps (config := .lemmasOnly) repr_symm_apply]
def basis : Basis ι R M :=
  Basis.ofRepr
    { toFun := coeffs h
      invFun := lc e
      left_inv := lc_coeffs h
      right_inv := coeffs_lc h
      map_add' := fun v w => by
        /-
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          e : ι → M
          ε : ι → Module.Dual R M
          h : Module.DualBases e ε
          v w : M
          ⊢ Eq (h.coeffs (HAdd.hAdd v w)) (HAdd.hAdd (h.coeffs v) (h.coeffs w))
        -/
        ext i
        /-
          case h
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          e : ι → M
          ε : ι → Module.Dual R M
          h : Module.DualBases e ε
          v w : M
          i : ι
          ⊢ Eq ((h.coeffs (HAdd.hAdd v w)) i) ((HAdd.hAdd (h.coeffs v) (h.coeffs w)) i)
        -/
        exact (ε i).map_add v w
        /-
          🎉 no goals
        -/
      map_smul' := fun c v => by
        /-
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          e : ι → M
          ε : ι → Module.Dual R M
          h : Module.DualBases e ε
          c : R
          v : M
          ⊢ Eq ({ toFun := h.coeffs, map_add' := ⋯ }.toFun (HSMul.hSMul c v)) (HSMul.hSM …
        -/
        ext i
        /-
          case h
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          e : ι → M
          ε : ι → Module.Dual R M
          h : Module.DualBases e ε
          c : R
          v : M
          i : ι
          ⊢ Eq (({ toFun := h.coeffs, map_add' := ⋯ }.toFun (HSMul.hSMul c v)) i) ((HSMu …
        -/
        exact (ε i).map_smul c v }
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_basis : ⇑h.basis = e := by
  /-
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    ⊢ Eq (⇑h.basis) e
  -/
  ext i
  /-
    case h
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    i : ι
    ⊢ Eq (h.basis i) (e i)
  -/
  rw [Basis.apply_eq_iff]
  /-
    case h
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    i : ι
    ⊢ Eq (h.basis.repr (e i)) (Finsupp.single i 1)
  -/
  ext j
  /-
    case h.h
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    i j : ι
    ⊢ Eq ((h.basis.repr (e i)) j) ((Finsupp.single i 1) j)
  -/
  rcases eq_or_ne i j with rfl | hne
    /-
      case h.h.inl
      R : Type u_1
      M : Type u_2
      ι : Type u_3
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      e : ι → M
      ε : ι → Module.Dual R M
      h : Module.DualBases e ε
      i : ι
      ⊢ Eq ((h.basis.repr (e i)) i) ((Finsupp.single i 1) i)
    -/
  · simp [h.eval_same]
    /-
      🎉 no goals
    -/
    /-
      case h.h.inr
      R : Type u_1
      M : Type u_2
      ι : Type u_3
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      e : ι → M
      ε : ι → Module.Dual R M
      h : Module.DualBases e ε
      i j : ι
      hne : Ne i j
      ⊢ Eq ((h.basis.repr (e i)) j) ((Finsupp.single i 1) j)
    -/
  · simp [hne, h.eval_of_ne hne.symm]
    /-
      🎉 no goals
    -/


theorem mem_of_mem_span {H : Set ι} {x : M} (hmem : x ∈ Submodule.span R (e '' H)) :
    ∀ i : ι, ε i x ≠ 0 → i ∈ H := by
  /-
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    H : Set ι
    x : M
    hmem : Membership.mem (Submodule.span R (Set.image e H)) x
    ⊢ ∀ (i : ι), Ne ((ε i) x) 0 → Membership.mem H i
  -/
  intro i hi
  /-
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    H : Set ι
    x : M
    hmem : Membership.mem (Submodule.span R (Set.image e H)) x
    i : ι
    hi : Ne ((ε i) x) 0
    ⊢ Membership.mem H i
  -/
  rcases (Finsupp.mem_span_image_iff_linearCombination _).mp hmem with ⟨l, supp_l, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    H : Set ι
    i : ι
    l : Finsupp ι R
    supp_l : Membership.mem (Finsupp.supported R R H) l
    hmem : Membership.mem (Submodule.span R (Set.image e H)) ((Finsupp.linearCombi …
    hi : Ne ((ε i) ((Finsupp.linearCombination R e) l)) 0
    ⊢ Membership.mem H i
  -/
  apply not_imp_comm.mp ((Finsupp.mem_supported' _ _).mp supp_l i)
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ι → M
    ε : ι → Module.Dual R M
    h : Module.DualBases e ε
    H : Set ι
    i : ι
    l : Finsupp ι R
    supp_l : Membership.mem (Finsupp.supported R R H) l
    hmem : Membership.mem (Submodule.span R (Set.image e H)) ((Finsupp.linearCombi …
    hi : Ne ((ε i) ((Finsupp.linearCombination R e) l)) 0
    ⊢ Not (Eq (l i) 0)
  -/
  rwa [← lc_def, h.dual_lc] at hi
  /-
    🎉 no goals
  -/


theorem coe_dualBasis [DecidableEq ι] [_root_.Finite ι] : ⇑h.basis.dualBasis = ε :=
                                          /-
                                            R : Type u_1
                                            M : Type u_2
                                            ι : Type u_3
                                            inst✝⁴ : CommRing R
                                            inst✝³ : AddCommGroup M
                                            inst✝² : Module R M
                                            e : ι → M
                                            ε : ι → Module.Dual R M
                                            h : Module.DualBases e ε
                                            inst✝¹ : DecidableEq ι
                                            inst✝ : Finite ι
                                            i j : ι
                                            ⊢ Eq ((h.basis.dualBasis i) (h.basis j)) ((ε i) (h.basis j))
                                          -/
  funext fun i => h.basis.ext fun j => by simp
                                          /-
                                            🎉 no goals
                                          -/


/-- The `dualRestrict` of a submodule `W` of `M` is the linear map from the
  dual of `M` to the dual of `W` such that the domain of each linear map is
  restricted to `W`. -/
def dualRestrict (W : Submodule R M) : Module.Dual R M →ₗ[R] Module.Dual R W :=
  LinearMap.domRestrict' W


theorem dualRestrict_def (W : Submodule R M) : W.dualRestrict = W.subtype.dualMap :=
  rfl


@[simp]
theorem dualRestrict_apply (W : Submodule R M) (φ : Module.Dual R M) (x : W) :
    W.dualRestrict φ x = φ (x : M) :=
  rfl


/-- The `dualAnnihilator` of a submodule `W` is the set of linear maps `φ` such
  that `φ w = 0` for all `w ∈ W`. -/
def dualAnnihilator {R : Type u} {M : Type v} [CommSemiring R] [AddCommMonoid M] [Module R M]
    (W : Submodule R M) : Submodule R <| Module.Dual R M :=
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629 LinearMap.ker
  LinearMap.ker W.dualRestrict


@[simp]
theorem mem_dualAnnihilator (φ : Module.Dual R M) : φ ∈ W.dualAnnihilator ↔ ∀ w ∈ W, φ w = 0 := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    W : Submodule R M
    φ : Module.Dual R M
    ⊢ Iff (Membership.mem W.dualAnnihilator φ) (∀ (w : M), Membership.mem W w → Eq …
  -/
  refine LinearMap.mem_ker.trans ?_
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    W : Submodule R M
    φ : Module.Dual R M
    ⊢ Iff (Eq (W.dualRestrict φ) 0) (∀ (w : M), Membership.mem W w → Eq (φ w) 0)
  -/
  simp_rw [LinearMap.ext_iff, dualRestrict_apply]
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    W : Submodule R M
    φ : Module.Dual R M
    ⊢ Iff (∀ (x : Subtype fun x => Membership.mem W x), Eq (φ ↑x) (0 x)) (∀ (w : M …
  -/
  exact ⟨fun h w hw => h ⟨w, hw⟩, fun h w => h w.1 w.2⟩
  /-
    🎉 no goals
  -/


/-- That $\operatorname{ker}(\iota^* : V^* \to W^*) = \operatorname{ann}(W)$.
This is the definition of the dual annihilator of the submodule $W$. -/
theorem dualRestrict_ker_eq_dualAnnihilator (W : Submodule R M) :
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629 LinearMap.ker
    LinearMap.ker W.dualRestrict = W.dualAnnihilator :=
  rfl


/-- The `dualAnnihilator` of a submodule of the dual space pulled back along the evaluation map
`Module.Dual.eval`. -/
def dualCoannihilator (Φ : Submodule R (Module.Dual R M)) : Submodule R M :=
  Φ.dualAnnihilator.comap (Module.Dual.eval R M)


@[simp]
theorem mem_dualCoannihilator {Φ : Submodule R (Module.Dual R M)} (x : M) :
    x ∈ Φ.dualCoannihilator ↔ ∀ φ ∈ Φ, (φ x : R) = 0 := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Submodule R (Module.Dual R M)
    x : M
    ⊢ Iff (Membership.mem Φ.dualCoannihilator x) (∀ (φ : Module.Dual R M), Members …
  -/
  simp_rw [dualCoannihilator, mem_comap, mem_dualAnnihilator, Module.Dual.eval_apply]
  /-
    🎉 no goals
  -/


theorem comap_dualAnnihilator (Φ : Submodule R (Module.Dual R M)) :
    Φ.dualAnnihilator.comap (Module.Dual.eval R M) = Φ.dualCoannihilator := rfl


theorem map_dualCoannihilator_le (Φ : Submodule R (Module.Dual R M)) :
    Φ.dualCoannihilator.map (Module.Dual.eval R M) ≤ Φ.dualAnnihilator :=
  map_le_iff_le_comap.mpr (comap_dualAnnihilator Φ).le


variable (R M) in
theorem dualAnnihilator_gc :
    GaloisConnection
      (OrderDual.toDual ∘ (dualAnnihilator : Submodule R M → Submodule R (Module.Dual R M)))
      (dualCoannihilator ∘ OrderDual.ofDual) := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ GaloisConnection (Function.comp (⇑OrderDual.toDual) Submodule.dualAnnihilato …
  -/
  intro a b
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a : Submodule R M
    b : OrderDual (Submodule R (Module.Dual R M))
    ⊢ Iff (LE.le (Function.comp (⇑OrderDual.toDual) Submodule.dualAnnihilator a) b …
  -/
  induction b using OrderDual.rec
  /-
    case h₂
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a : Submodule R M
    a✝ : Submodule R (Module.Dual R M)
    ⊢ Iff (LE.le (Function.comp (⇑OrderDual.toDual) Submodule.dualAnnihilator a) ( …
  -/
  simp only [Function.comp_apply, OrderDual.toDual_le_toDual, OrderDual.ofDual_toDual]
  /-
    case h₂
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a : Submodule R M
    a✝ : Submodule R (Module.Dual R M)
    ⊢ Iff (LE.le a✝ a.dualAnnihilator) (LE.le a a✝.dualCoannihilator)
  -/
  constructor <;>
      /-
        case h₂.mp
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        a : Submodule R M
        a✝ : Submodule R (Module.Dual R M)
        ⊢ LE.le a✝ a.dualAnnihilator → LE.le a a✝.dualCoannihilator
      -/
      /-
        case h₂.mp
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        a : Submodule R M
        a✝ : Submodule R (Module.Dual R M)
        h : LE.le a✝ a.dualAnnihilator
        x : M
        hx : Membership.mem a x
        ⊢ Membership.mem a✝.dualCoannihilator x
      -/
      /-
        case h₂.mp
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        a : Submodule R M
        a✝ : Submodule R (Module.Dual R M)
        h : LE.le a✝ a.dualAnnihilator
        x : M
        hx : Membership.mem a x
        ⊢ ∀ (φ : Module.Dual R M), Membership.mem a✝ φ → Eq (φ x) 0
      -/
      /-
        case h₂.mp
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        a : Submodule R M
        a✝ : Submodule R (Module.Dual R M)
        h : LE.le a✝ a.dualAnnihilator
        x : M
        hx : Membership.mem a x
        y : Module.Dual R M
        hy : Membership.mem a✝ y
        ⊢ Eq (y x) 0
      -/
      /-
        case h₂.mp
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        a : Submodule R M
        a✝ : Submodule R (Module.Dual R M)
        h : LE.le a✝ a.dualAnnihilator
        x : M
        hx : Membership.mem a x
        y : Module.Dual R M
        hy : Membership.mem a✝ y
        this : Membership.mem a.dualAnnihilator y
        ⊢ Eq (y x) 0
      -/
      /-
        case h₂.mp
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        a : Submodule R M
        a✝ : Submodule R (Module.Dual R M)
        h : LE.le a✝ a.dualAnnihilator
        x : M
        hx : Membership.mem a x
        y : Module.Dual R M
        hy : Membership.mem a✝ y
        this : ∀ (w : M), Membership.mem a w → Eq (y w) 0
        ⊢ Eq (y x) 0
      -/
      /-
        🎉 no goals
      -/
      /-
        case h₂.mpr
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        a : Submodule R M
        a✝ : Submodule R (Module.Dual R M)
        h : LE.le a a✝.dualCoannihilator
        x : Module.Dual R M
        hx : Membership.mem a✝ x
        y : M
        hy : Membership.mem a y
        ⊢ Eq (x y) 0
      -/
      have := h hy
      /-
        case h₂.mpr
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        a : Submodule R M
        a✝ : Submodule R (Module.Dual R M)
        h : LE.le a a✝.dualCoannihilator
        x : Module.Dual R M
        hx : Membership.mem a✝ x
        y : M
        hy : Membership.mem a y
        this : Membership.mem a✝.dualCoannihilator y
        ⊢ Eq (x y) 0
      -/
      simp only [mem_dualAnnihilator, mem_dualCoannihilator] at this
      /-
        case h₂.mpr
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        a : Submodule R M
        a✝ : Submodule R (Module.Dual R M)
        h : LE.le a a✝.dualCoannihilator
        x : Module.Dual R M
        hx : Membership.mem a✝ x
        y : M
        hy : Membership.mem a y
        this : ∀ (φ : Module.Dual R M), Membership.mem a✝ φ → Eq (φ y) 0
        ⊢ Eq (x y) 0
      -/
      exact this x hx
      /-
        🎉 no goals
      -/


theorem le_dualAnnihilator_iff_le_dualCoannihilator {U : Submodule R (Module.Dual R M)}
    {V : Submodule R M} : U ≤ V.dualAnnihilator ↔ V ≤ U.dualCoannihilator :=
  (dualAnnihilator_gc R M).le_iff_le


@[simp]
theorem dualAnnihilator_bot : (⊥ : Submodule R M).dualAnnihilator = ⊤ :=
  (dualAnnihilator_gc R M).l_bot


@[simp]
theorem dualAnnihilator_top : (⊤ : Submodule R M).dualAnnihilator = ⊥ := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Eq Top.top.dualAnnihilator Bot.bot
  -/
  rw [eq_bot_iff]
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ LE.le Top.top.dualAnnihilator Bot.bot
  -/
  intro v
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : Module.Dual R M
    ⊢ Membership.mem Top.top.dualAnnihilator v → Membership.mem Bot.bot v
  -/
  simp_rw [mem_dualAnnihilator, mem_bot, mem_top, forall_true_left]
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    v : Module.Dual R M
    ⊢ (∀ (w : M), Eq (v w) 0) → Eq v 0
  -/
  exact fun h => LinearMap.ext h
  /-
    🎉 no goals
  -/


@[simp]
theorem dualCoannihilator_bot : (⊥ : Submodule R (Module.Dual R M)).dualCoannihilator = ⊤ :=
  (dualAnnihilator_gc R M).u_top


@[mono]
theorem dualAnnihilator_anti {U V : Submodule R M} (hUV : U ≤ V) :
    V.dualAnnihilator ≤ U.dualAnnihilator :=
  (dualAnnihilator_gc R M).monotone_l hUV


@[mono]
theorem dualCoannihilator_anti {U V : Submodule R (Module.Dual R M)} (hUV : U ≤ V) :
    V.dualCoannihilator ≤ U.dualCoannihilator :=
  (dualAnnihilator_gc R M).monotone_u hUV


theorem le_dualAnnihilator_dualCoannihilator (U : Submodule R M) :
    U ≤ U.dualAnnihilator.dualCoannihilator :=
  (dualAnnihilator_gc R M).le_u_l U


theorem le_dualCoannihilator_dualAnnihilator (U : Submodule R (Module.Dual R M)) :
    U ≤ U.dualCoannihilator.dualAnnihilator :=
  (dualAnnihilator_gc R M).l_u_le U


theorem dualAnnihilator_dualCoannihilator_dualAnnihilator (U : Submodule R M) :
    U.dualAnnihilator.dualCoannihilator.dualAnnihilator = U.dualAnnihilator :=
  (dualAnnihilator_gc R M).l_u_l_eq_l U


theorem dualCoannihilator_dualAnnihilator_dualCoannihilator (U : Submodule R (Module.Dual R M)) :
    U.dualCoannihilator.dualAnnihilator.dualCoannihilator = U.dualCoannihilator :=
  (dualAnnihilator_gc R M).u_l_u_eq_u U


theorem dualAnnihilator_sup_eq (U V : Submodule R M) :
    (U ⊔ V).dualAnnihilator = U.dualAnnihilator ⊓ V.dualAnnihilator :=
  (dualAnnihilator_gc R M).l_sup


theorem dualCoannihilator_sup_eq (U V : Submodule R (Module.Dual R M)) :
    (U ⊔ V).dualCoannihilator = U.dualCoannihilator ⊓ V.dualCoannihilator :=
  (dualAnnihilator_gc R M).u_inf


theorem dualAnnihilator_iSup_eq {ι : Sort*} (U : ι → Submodule R M) :
    (⨆ i : ι, U i).dualAnnihilator = ⨅ i : ι, (U i).dualAnnihilator :=
  (dualAnnihilator_gc R M).l_iSup


theorem dualCoannihilator_iSup_eq {ι : Sort*} (U : ι → Submodule R (Module.Dual R M)) :
    (⨆ i : ι, U i).dualCoannihilator = ⨅ i : ι, (U i).dualCoannihilator :=
  (dualAnnihilator_gc R M).u_iInf


/-- See also `Subspace.dualAnnihilator_inf_eq` for vector subspaces. -/
theorem sup_dualAnnihilator_le_inf (U V : Submodule R M) :
    U.dualAnnihilator ⊔ V.dualAnnihilator ≤ (U ⊓ V).dualAnnihilator := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    U V : Submodule R M
    ⊢ LE.le (Max.max U.dualAnnihilator V.dualAnnihilator) (Min.min U V).dualAnnihi …
  -/
  rw [le_dualAnnihilator_iff_le_dualCoannihilator, dualCoannihilator_sup_eq]
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    U V : Submodule R M
    ⊢ LE.le (Min.min U V) (Min.min U.dualAnnihilator.dualCoannihilator V.dualAnnih …
  -/
                       /-
                         🎉 no goals
                       -/
  apply inf_le_inf <;> exact le_dualAnnihilator_dualCoannihilator _
                       /-
                         🎉 no goals
                       -/


/-- See also `Subspace.dualAnnihilator_iInf_eq` for vector subspaces when `ι` is finite. -/
theorem iSup_dualAnnihilator_le_iInf {ι : Sort*} (U : ι → Submodule R M) :
    ⨆ i : ι, (U i).dualAnnihilator ≤ (⨅ i : ι, U i).dualAnnihilator := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Sort u_1
    U : ι → Submodule R M
    ⊢ LE.le (iSup fun i => (U i).dualAnnihilator) (iInf fun i => U i).dualAnnihila …
  -/
  rw [le_dualAnnihilator_iff_le_dualCoannihilator, dualCoannihilator_iSup_eq]
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Sort u_1
    U : ι → Submodule R M
    ⊢ LE.le (iInf fun i => U i) (iInf fun i => (U i).dualAnnihilator.dualCoannihil …
  -/
  apply iInf_mono
  /-
    case h
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Sort u_1
    U : ι → Submodule R M
    ⊢ ∀ (i : ι), LE.le (U i) (U i).dualAnnihilator.dualCoannihilator
  -/
  exact fun i : ι => le_dualAnnihilator_dualCoannihilator (U i)
  /-
    🎉 no goals
  -/


@[simp]
lemma coe_dualAnnihilator_span (s : Set M) :
    ((span R s).dualAnnihilator : Set (Module.Dual R M)) = {f | s ⊆ LinearMap.ker f} := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set M
    ⊢ Eq (↑(Submodule.span R s).dualAnnihilator) (setOf fun f => HasSubset.Subset  …
  -/
  ext f
  /-
    case h
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set M
    f : Module.Dual R M
    ⊢ Iff (Membership.mem (↑(Submodule.span R s).dualAnnihilator) f) (Membership.m …
  -/
  simp only [SetLike.mem_coe, mem_dualAnnihilator, Set.mem_setOf_eq, ← LinearMap.mem_ker]
  /-
    case h
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set M
    f : Module.Dual R M
    ⊢ Iff (∀ (w : M), Membership.mem (Submodule.span R s) w → Membership.mem (Line …
  -/
  exact span_le
  /-
    🎉 no goals
  -/


@[simp]
lemma coe_dualCoannihilator_span (s : Set (Module.Dual R M)) :
    ((span R s).dualCoannihilator : Set M) = {x | ∀ f ∈ s, f x = 0} := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set (Module.Dual R M)
    ⊢ Eq (↑(Submodule.span R s).dualCoannihilator) (setOf fun x => ∀ (f : Module.D …
  -/
  ext x
  /-
    case h
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set (Module.Dual R M)
    x : M
    ⊢ Iff (Membership.mem (↑(Submodule.span R s).dualCoannihilator) x) (Membership …
  -/
  have (φ) : x ∈ LinearMap.ker φ ↔ φ ∈ LinearMap.ker (Module.Dual.eval R M x) := by simp
  /-
    case h
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set (Module.Dual R M)
    x : M
    this : ∀ (φ : Module.Dual R M), Iff (Membership.mem (LinearMap.ker φ) x) (Memb …
    ⊢ Iff (Membership.mem (↑(Submodule.span R s).dualCoannihilator) x) (Membership …
  -/
  simp only [SetLike.mem_coe, mem_dualCoannihilator, Set.mem_setOf_eq, ← LinearMap.mem_ker, this]
  /-
    case h
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set (Module.Dual R M)
    x : M
    this : ∀ (φ : Module.Dual R M), Iff (Membership.mem (LinearMap.ker φ) x) (Memb …
    ⊢ Iff (∀ (φ : Module.Dual R M), Membership.mem (Submodule.span R s) φ → Member …
  -/
  exact span_le
  /-
    🎉 no goals
  -/


@[simp]
theorem dualCoannihilator_top (W : Subspace K V) :
    (⊤ : Subspace K (Module.Dual K W)).dualCoannihilator = ⊥ := by
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    ⊢ Eq (Submodule.dualCoannihilator Top.top) Bot.bot
  -/
  rw [dualCoannihilator, dualAnnihilator_top, comap_bot, Module.eval_ker]
  /-
    🎉 no goals
  -/


@[simp]
theorem dualAnnihilator_dualCoannihilator_eq {W : Subspace K V} :
    W.dualAnnihilator.dualCoannihilator = W := by
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    ⊢ Eq (Submodule.dualAnnihilator W).dualCoannihilator W
  -/
  refine le_antisymm (fun v ↦ Function.mtr ?_) (le_dualAnnihilator_dualCoannihilator _)
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    v : V
    ⊢ Not (Membership.mem W v) → Not (Membership.mem (Submodule.dualAnnihilator W) …
  -/
  simp only [mem_dualAnnihilator, mem_dualCoannihilator]
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    v : V
    ⊢ Not (Membership.mem W v) → Not (∀ (φ : Module.Dual K V), (∀ (w : V), Members …
  -/
  rw [← Quotient.mk_eq_zero W, ← Module.forall_dual_apply_eq_zero_iff K]
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    v : V
    ⊢ Not (∀ (φ : Module.Dual K (HasQuotient.Quotient V W)), Eq (φ (Submodule.Quot …
  -/
  push_neg
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    v : V
    ⊢ (Exists fun φ => Ne (φ (Submodule.Quotient.mk v)) 0) → Exists fun φ => And ( …
  -/
  refine fun ⟨φ, hφ⟩ ↦ ⟨φ.comp W.mkQ, fun w hw ↦ ?_, hφ⟩
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    v : V
    x✝ : Exists fun φ => Ne (φ (Submodule.Quotient.mk v)) 0
    φ : Module.Dual K (HasQuotient.Quotient V W)
    hφ : Ne (φ (Submodule.Quotient.mk v)) 0
    w : V
    hw : Membership.mem W w
    ⊢ Eq ((LinearMap.comp φ (Submodule.mkQ W)) w) 0
  -/
  rw [comp_apply, mkQ_apply, (Quotient.mk_eq_zero W).mpr hw, φ.map_zero]
  /-
    🎉 no goals
  -/

-- exact elaborates slowly

theorem forall_mem_dualAnnihilator_apply_eq_zero_iff (W : Subspace K V) (v : V) :
    (∀ φ : Module.Dual K V, φ ∈ W.dualAnnihilator → φ v = 0) ↔ v ∈ W := by
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    v : V
    ⊢ Iff (∀ (φ : Module.Dual K V), Membership.mem (Submodule.dualAnnihilator W) φ …
  -/
  rw [← SetLike.ext_iff.mp dualAnnihilator_dualCoannihilator_eq v, mem_dualCoannihilator]
  /-
    🎉 no goals
  -/


theorem comap_dualAnnihilator_dualAnnihilator (W : Subspace K V) :
    W.dualAnnihilator.dualAnnihilator.comap (Module.Dual.eval K V) = W := by
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    ⊢ Eq (Submodule.comap (Module.Dual.eval K V) (Submodule.dualAnnihilator W).dua …
  -/
  ext; rw [Iff.comm, ← forall_mem_dualAnnihilator_apply_eq_zero_iff]; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem map_le_dualAnnihilator_dualAnnihilator (W : Subspace K V) :
    W.map (Module.Dual.eval K V) ≤ W.dualAnnihilator.dualAnnihilator :=
  map_le_iff_le_comap.mpr (comap_dualAnnihilator_dualAnnihilator W).ge


/-- `Submodule.dualAnnihilator` and `Submodule.dualCoannihilator` form a Galois coinsertion. -/
def dualAnnihilatorGci (K V : Type*) [Field K] [AddCommGroup V] [Module K V] :
    GaloisCoinsertion
      (OrderDual.toDual ∘ (dualAnnihilator : Subspace K V → Subspace K (Module.Dual K V)))
      (dualCoannihilator ∘ OrderDual.ofDual) where
  choice W _ := dualCoannihilator W
  gc := dualAnnihilator_gc K V
  u_l_le _ := dualAnnihilator_dualCoannihilator_eq.le
  choice_eq _ _ := rfl


theorem dualAnnihilator_le_dualAnnihilator_iff {W W' : Subspace K V} :
    W.dualAnnihilator ≤ W'.dualAnnihilator ↔ W' ≤ W :=
  (dualAnnihilatorGci K V).l_le_l_iff


theorem dualAnnihilator_inj {W W' : Subspace K V} :
    W.dualAnnihilator = W'.dualAnnihilator ↔ W = W' :=
  ⟨fun h ↦ (dualAnnihilatorGci K V).l_injective h, congr_arg _⟩


/-- Given a subspace `W` of `V` and an element of its dual `φ`, `dualLift W φ` is
an arbitrary extension of `φ` to an element of the dual of `V`.
That is, `dualLift W φ` sends `w ∈ W` to `φ x` and `x` in a chosen complement of `W` to `0`. -/
noncomputable def dualLift (W : Subspace K V) : Module.Dual K W →ₗ[K] Module.Dual K V :=
  (Classical.choose <| W.subtype.exists_leftInverse_of_injective W.ker_subtype).dualMap


@[simp]
theorem dualLift_of_subtype {φ : Module.Dual K W} (w : W) : W.dualLift φ (w : V) = φ w :=
  congr_arg φ <| DFunLike.congr_fun
    (Classical.choose_spec <| W.subtype.exists_leftInverse_of_injective W.ker_subtype) w


theorem dualLift_of_mem {φ : Module.Dual K W} {w : V} (hw : w ∈ W) : W.dualLift φ w = φ ⟨w, hw⟩ :=
  dualLift_of_subtype ⟨w, hw⟩


@[simp]
theorem dualRestrict_comp_dualLift (W : Subspace K V) : W.dualRestrict.comp W.dualLift = 1 := by
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    ⊢ Eq ((Submodule.dualRestrict W).comp W.dualLift) 1
  -/
  ext φ x
  /-
    case h.h
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    φ : Module.Dual K (Subtype fun x => Membership.mem W x)
    x : Subtype fun x => Membership.mem W x
    ⊢ Eq ((((Submodule.dualRestrict W).comp W.dualLift) φ) x) ((1 φ) x)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem dualRestrict_leftInverse (W : Subspace K V) :
    Function.LeftInverse W.dualRestrict W.dualLift := fun x =>
  show W.dualRestrict.comp W.dualLift x = x by
    /-
      K : Type u
      V : Type v
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      W : Subspace K V
      x : Module.Dual K (Subtype fun x => Membership.mem W x)
      ⊢ Eq (((Submodule.dualRestrict W).comp W.dualLift) x) x
    -/
    rw [dualRestrict_comp_dualLift]
    /-
      K : Type u
      V : Type v
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      W : Subspace K V
      x : Module.Dual K (Subtype fun x => Membership.mem W x)
      ⊢ Eq (1 x) x
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem dualLift_rightInverse (W : Subspace K V) :
    Function.RightInverse W.dualLift W.dualRestrict :=
  W.dualRestrict_leftInverse


theorem dualRestrict_surjective : Function.Surjective W.dualRestrict :=
  W.dualLift_rightInverse.surjective


theorem dualLift_injective : Function.Injective W.dualLift :=
  W.dualRestrict_leftInverse.injective


/-- The quotient by the `dualAnnihilator` of a subspace is isomorphic to the
  dual of that subspace. -/
noncomputable def quotAnnihilatorEquiv (W : Subspace K V) :
    (Module.Dual K V ⧸ W.dualAnnihilator) ≃ₗ[K] Module.Dual K W :=
  (quotEquivOfEq _ _ W.dualRestrict_ker_eq_dualAnnihilator).symm.trans <|
    W.dualRestrict.quotKerEquivOfSurjective dualRestrict_surjective


@[simp]
theorem quotAnnihilatorEquiv_apply (W : Subspace K V) (φ : Module.Dual K V) :
    W.quotAnnihilatorEquiv (Submodule.Quotient.mk φ) = W.dualRestrict φ := by
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    φ : Module.Dual K V
    ⊢ Eq (W.quotAnnihilatorEquiv (Submodule.Quotient.mk φ)) ((Submodule.dualRestri …
  -/
  ext
  /-
    case h
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Subspace K V
    φ : Module.Dual K V
    x✝ : Subtype fun x => Membership.mem W x
    ⊢ Eq ((W.quotAnnihilatorEquiv (Submodule.Quotient.mk φ)) x✝) (((Submodule.dual …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The natural isomorphism from the dual of a subspace `W` to `W.dualLift.range`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629 LinearMap.range
noncomputable def dualEquivDual (W : Subspace K V) :
    Module.Dual K W ≃ₗ[K] LinearMap.range W.dualLift :=
  LinearEquiv.ofInjective _ dualLift_injective


theorem dualEquivDual_def (W : Subspace K V) :
    W.dualEquivDual.toLinearMap = W.dualLift.rangeRestrict :=
  rfl


@[simp]
theorem dualEquivDual_apply (φ : Module.Dual K W) :
    W.dualEquivDual φ = ⟨W.dualLift φ, mem_range.2 ⟨φ, rfl⟩⟩ :=
  rfl


instance instModuleDualFiniteDimensional [FiniteDimensional K V] :
    FiniteDimensional K (Module.Dual K V) := by
  /-
    K : Type u
    V : Type v
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    W : Subspace K V
    inst✝ : FiniteDimensional K V
    ⊢ FiniteDimensional K (Module.Dual K V)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem dual_finrank_eq : finrank K (Module.Dual K V) = finrank K V := by
  /-
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ⊢ Eq (Module.finrank K (Module.Dual K V)) (Module.finrank K V)
  -/
  by_cases h : FiniteDimensional K V
    /-
      case pos
      K : Type u
      V : Type v
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      h : FiniteDimensional K V
      ⊢ Eq (Module.finrank K (Module.Dual K V)) (Module.finrank K V)
    -/
  · classical exact LinearEquiv.finrank_eq (Basis.ofVectorSpace K V).toDualEquiv.symm
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u
    V : Type v
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : Not (FiniteDimensional K V)
    ⊢ Eq (Module.finrank K (Module.Dual K V)) (Module.finrank K V)
  -/
  rw [finrank_eq_zero_of_basis_imp_false, finrank_eq_zero_of_basis_imp_false]
    /-
      case neg
      K : Type u
      V : Type v
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      h : Not (FiniteDimensional K V)
      ⊢ ∀ (s : Finset V), Basis (↑↑s) K V → False
    -/
  · exact fun _ b ↦ h (Module.Finite.of_basis b)
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u
      V : Type v
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      h : Not (FiniteDimensional K V)
      ⊢ ∀ (s : Finset (Module.Dual K V)), Basis (↑↑s) K (Module.Dual K V) → False
    -/
  · exact fun _ b ↦ h ((Module.finite_dual_iff K).mp <| Module.Finite.of_basis b)
    /-
      🎉 no goals
    -/


theorem dualAnnihilator_dualAnnihilator_eq (W : Subspace K V) :
    W.dualAnnihilator.dualAnnihilator = Module.mapEvalEquiv K V W := by
  /-
    K : Type u
    V : Type v
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    W : Subspace K V
    ⊢ Eq (Submodule.dualAnnihilator W).dualAnnihilator ((Module.mapEvalEquiv K V) W)
  -/
  have : _ = W := Subspace.dualAnnihilator_dualCoannihilator_eq
  /-
    K : Type u
    V : Type v
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    W : Subspace K V
    this : Eq (Submodule.dualAnnihilator W).dualCoannihilator W
    ⊢ Eq (Submodule.dualAnnihilator W).dualAnnihilator ((Module.mapEvalEquiv K V) W)
  -/
  rw [dualCoannihilator, ← Module.mapEvalEquiv_symm_apply] at this
  /-
    K : Type u
    V : Type v
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    W : Subspace K V
    this : Eq ((Module.mapEvalEquiv K V).symm (Submodule.dualAnnihilator W).dualAn …
    ⊢ Eq (Submodule.dualAnnihilator W).dualAnnihilator ((Module.mapEvalEquiv K V) W)
  -/
  rwa [← OrderIso.symm_apply_eq]
  /-
    🎉 no goals
  -/


/-- The quotient by the dual is isomorphic to its dual annihilator. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629 LinearMap.range
noncomputable def quotDualEquivAnnihilator (W : Subspace K V) :
    (Module.Dual K V ⧸ LinearMap.range W.dualLift) ≃ₗ[K] W.dualAnnihilator :=
  LinearEquiv.quotEquivOfQuotEquiv <| LinearEquiv.trans W.quotAnnihilatorEquiv W.dualEquivDual


open scoped Classical in
/-- The quotient by a subspace is isomorphic to its dual annihilator. -/
noncomputable def quotEquivAnnihilator (W : Subspace K V) : (V ⧸ W) ≃ₗ[K] W.dualAnnihilator :=
  let φ := (Basis.ofVectorSpace K W).toDualEquiv.trans W.dualEquivDual
  let ψ := LinearEquiv.quotEquivOfEquiv φ (Basis.ofVectorSpace K V).toDualEquiv
  ψ ≪≫ₗ W.quotDualEquivAnnihilator
  -- Porting note: this prevents the timeout; ML3 proof preserved below
  -- refine' _ ≪≫ₗ W.quotDualEquivAnnihilator
  -- refine' LinearEquiv.quot_equiv_of_equiv _ (Basis.ofVectorSpace K V).toDualEquiv
  -- exact (Basis.ofVectorSpace K W).toDualEquiv.trans W.dual_equiv_dual


theorem finrank_add_finrank_dualAnnihilator_eq (W : Subspace K V) :
    finrank K W + finrank K W.dualAnnihilator = finrank K V := by
  rw [← W.quotEquivAnnihilator.finrank_eq (M₂ := dualAnnihilator W),
    add_comm, Submodule.finrank_quotient_add_finrank]


@[simp]
theorem finrank_dualCoannihilator_eq {Φ : Subspace K (Module.Dual K V)} :
    finrank K Φ.dualCoannihilator = finrank K Φ.dualAnnihilator := by
  /-
    K : Type u
    V : Type v
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    Φ : Subspace K (Module.Dual K V)
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (Submodule.dualCoannih …
  -/
  rw [Submodule.dualCoannihilator, ← Module.evalEquiv_toLinearMap]
  /-
    K : Type u
    V : Type v
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    Φ : Subspace K (Module.Dual K V)
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (Submodule.comap (↑(Mo …
  -/
  exact LinearEquiv.finrank_eq (LinearEquiv.ofSubmodule' _ _)
  /-
    🎉 no goals
  -/


theorem finrank_add_finrank_dualCoannihilator_eq (W : Subspace K (Module.Dual K V)) :
    finrank K W + finrank K W.dualCoannihilator = finrank K V := by
  /-
    K : Type u
    V : Type v
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    W : Subspace K (Module.Dual K V)
    ⊢ Eq (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem W x)) (Modu …
  -/
  rw [finrank_dualCoannihilator_eq, finrank_add_finrank_dualAnnihilator_eq, dual_finrank_eq]
  /-
    🎉 no goals
  -/


theorem ker_dualMap_eq_dualAnnihilator_range :
    LinearMap.ker f.dualMap = f.range.dualAnnihilator := by
  /-
    R : Type uR
    inst✝⁴ : CommSemiring R
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    ⊢ Eq (LinearMap.ker f.dualMap) (LinearMap.range f).dualAnnihilator
  -/
  ext
  simp_rw [mem_ker, LinearMap.ext_iff, Submodule.mem_dualAnnihilator,
    ← SetLike.mem_coe, range_coe, Set.forall_mem_range]
  /-
    case h
    R : Type uR
    inst✝⁴ : CommSemiring R
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    x✝ : Module.Dual R M₂
    ⊢ Iff (∀ (x : M₁), Eq ((f.dualMap x✝) x) (0 x)) (∀ (i : M₁), Eq (x✝ (f i)) 0)
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629 LinearMap.range

theorem range_dualMap_le_dualAnnihilator_ker :
    LinearMap.range f.dualMap ≤ f.ker.dualAnnihilator := by
  /-
    R : Type uR
    inst✝⁴ : CommSemiring R
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    ⊢ LE.le (LinearMap.range f.dualMap) (LinearMap.ker f).dualAnnihilator
  -/
  rintro _ ⟨ψ, rfl⟩
  /-
    case intro
    R : Type uR
    inst✝⁴ : CommSemiring R
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    ψ : Module.Dual R M₂
    ⊢ Membership.mem (LinearMap.ker f).dualAnnihilator (f.dualMap ψ)
  -/
  simp_rw [Submodule.mem_dualAnnihilator, mem_ker]
  /-
    case intro
    R : Type uR
    inst✝⁴ : CommSemiring R
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    ψ : Module.Dual R M₂
    ⊢ ∀ (w : M₁), Eq (f w) 0 → Eq ((f.dualMap ψ) w) 0
  -/
  rintro x hx
  /-
    case intro
    R : Type uR
    inst✝⁴ : CommSemiring R
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    ψ : Module.Dual R M₂
    x : M₁
    hx : Eq (f x) 0
    ⊢ Eq ((f.dualMap ψ) x) 0
  -/
  rw [dualMap_apply, hx, map_zero]
  /-
    🎉 no goals
  -/


/-- Given a submodule, corestrict to the pairing on `M ⧸ W` by
simultaneously restricting to `W.dualAnnihilator`.

See `Subspace.dualCopairing_nondegenerate`. -/
def dualCopairing (W : Submodule R M) : W.dualAnnihilator →ₗ[R] M ⧸ W →ₗ[R] R :=
  LinearMap.flip <|
    W.liftQ ((Module.dualPairing R M).domRestrict W.dualAnnihilator).flip
      (by
        /-
          R : Type u_1
          M : Type u_2
          M' : Type u_3
          inst✝⁴ : CommRing R
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : AddCommGroup M'
          inst✝ : Module R M'
          W : Submodule R M
          ⊢ LE.le W (LinearMap.ker ((Module.dualPairing R M).domRestrict W.dualAnnihilat …
        -/
        intro w hw
        /-
          R : Type u_1
          M : Type u_2
          M' : Type u_3
          inst✝⁴ : CommRing R
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : AddCommGroup M'
          inst✝ : Module R M'
          W : Submodule R M
          w : M
          hw : Membership.mem W w
          ⊢ Membership.mem (LinearMap.ker ((Module.dualPairing R M).domRestrict W.dualAn …
        -/
        ext ⟨φ, hφ⟩
        /-
          case h.mk
          R : Type u_1
          M : Type u_2
          M' : Type u_3
          inst✝⁴ : CommRing R
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : AddCommGroup M'
          inst✝ : Module R M'
          W : Submodule R M
          w : M
          hw : Membership.mem W w
          φ : Module.Dual R M
          hφ : Membership.mem W.dualAnnihilator φ
          ⊢ Eq ((((Module.dualPairing R M).domRestrict W.dualAnnihilator).flip w) ⟨φ, hφ …
        -/
        exact (mem_dualAnnihilator φ).mp hφ w hw)
        /-
          🎉 no goals
        -/

-- Porting note: helper instance

instance (W : Submodule R M) : FunLike (W.dualAnnihilator) M R where
  coe φ := φ.val
  coe_injective' φ ψ h := by
    /-
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      W : Submodule R M
      φ ψ : Subtype fun x => Membership.mem W.dualAnnihilator x
      h : Eq ((fun φ => ⇑↑φ) φ) ((fun φ => ⇑↑φ) ψ)
      ⊢ Eq φ ψ
    -/
    ext
    /-
      case a.h
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      W : Submodule R M
      φ ψ : Subtype fun x => Membership.mem W.dualAnnihilator x
      h : Eq ((fun φ => ⇑↑φ) φ) ((fun φ => ⇑↑φ) ψ)
      x✝ : M
      ⊢ Eq (↑φ x✝) (↑ψ x✝)
    -/
    simp only [funext_iff] at h
    /-
      case a.h
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      W : Submodule R M
      φ ψ : Subtype fun x => Membership.mem W.dualAnnihilator x
      x✝ : M
      h : ∀ (x : M), Eq (↑φ x) (↑ψ x)
      ⊢ Eq (↑φ x✝) (↑ψ x✝)
    -/
    exact h _
    /-
      🎉 no goals
    -/


@[simp]
theorem dualCopairing_apply {W : Submodule R M} (φ : W.dualAnnihilator) (x : M) :
    W.dualCopairing φ (Quotient.mk x) = φ x :=
  rfl


/-- Given a submodule, restrict to the pairing on `W` by
simultaneously corestricting to `Module.Dual R M ⧸ W.dualAnnihilator`.
This is `Submodule.dualRestrict` factored through the quotient by its kernel (which
is `W.dualAnnihilator` by definition).

See `Subspace.dualPairing_nondegenerate`. -/
def dualPairing (W : Submodule R M) : Module.Dual R M ⧸ W.dualAnnihilator →ₗ[R] W →ₗ[R] R :=
  W.dualAnnihilator.liftQ W.dualRestrict le_rfl


@[simp]
theorem dualPairing_apply {W : Submodule R M} (φ : Module.Dual R M) (x : W) :
    W.dualPairing (Quotient.mk φ) x = φ x :=
  rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629 LinearMap.range

/-- That $\operatorname{im}(q^* : (V/W)^* \to V^*) = \operatorname{ann}(W)$. -/
theorem range_dualMap_mkQ_eq (W : Submodule R M) :
    LinearMap.range W.mkQ.dualMap = W.dualAnnihilator := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    W : Submodule R M
    ⊢ Eq (LinearMap.range W.mkQ.dualMap) W.dualAnnihilator
  -/
  ext φ
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    W : Submodule R M
    φ : Module.Dual R M
    ⊢ Iff (Membership.mem (LinearMap.range W.mkQ.dualMap) φ) (Membership.mem W.dua …
  -/
  rw [LinearMap.mem_range]
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    W : Submodule R M
    φ : Module.Dual R M
    ⊢ Iff (Exists fun y => Eq (W.mkQ.dualMap y) φ) (Membership.mem W.dualAnnihilat …
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      W : Submodule R M
      φ : Module.Dual R M
      ⊢ (Exists fun y => Eq (W.mkQ.dualMap y) φ) → Membership.mem W.dualAnnihilator φ
    -/
  · rintro ⟨ψ, rfl⟩
    /-
      case h.mp.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      W : Submodule R M
      ψ : Module.Dual R (HasQuotient.Quotient M W)
      ⊢ Membership.mem W.dualAnnihilator (W.mkQ.dualMap ψ)
    -/
    have := LinearMap.mem_range_self W.mkQ.dualMap ψ
    /-
      case h.mp.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      W : Submodule R M
      ψ : Module.Dual R (HasQuotient.Quotient M W)
      this : Membership.mem (LinearMap.range W.mkQ.dualMap) (W.mkQ.dualMap ψ)
      ⊢ Membership.mem W.dualAnnihilator (W.mkQ.dualMap ψ)
    -/
    simpa only [ker_mkQ] using W.mkQ.range_dualMap_le_dualAnnihilator_ker this
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      W : Submodule R M
      φ : Module.Dual R M
      ⊢ Membership.mem W.dualAnnihilator φ → Exists fun y => Eq (W.mkQ.dualMap y) φ
    -/
  · intro hφ
    /-
      case h.mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      W : Submodule R M
      φ : Module.Dual R M
      hφ : Membership.mem W.dualAnnihilator φ
      ⊢ Exists fun y => Eq (W.mkQ.dualMap y) φ
    -/
    exists W.dualCopairing ⟨φ, hφ⟩
    /-
      🎉 no goals
    -/


/-- Equivalence $(M/W)^* \cong \operatorname{ann}(W)$. That is, there is a one-to-one
correspondence between the dual of `M ⧸ W` and those elements of the dual of `M` that
vanish on `W`.

The inverse of this is `Submodule.dualCopairing`. -/
def dualQuotEquivDualAnnihilator (W : Submodule R M) :
    Module.Dual R (M ⧸ W) ≃ₗ[R] W.dualAnnihilator :=
  LinearEquiv.ofLinear
    (W.mkQ.dualMap.codRestrict W.dualAnnihilator fun φ =>
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629 LinearMap.mem_range_self
      W.range_dualMap_mkQ_eq ▸ LinearMap.mem_range_self W.mkQ.dualMap φ)
                        /-
                          R : Type u_1
                          M : Type u_2
                          M' : Type u_3
                          inst✝⁴ : CommRing R
                          inst✝³ : AddCommGroup M
                          inst✝² : Module R M
                          inst✝¹ : AddCommGroup M'
                          inst✝ : Module R M'
                          W : Submodule R M
                          ⊢ Eq ((LinearMap.codRestrict W.dualAnnihilator W.mkQ.dualMap ⋯).comp W.dualCop …
                        -/
                             /-
                               🎉 no goals
                             -/
    W.dualCopairing (by ext; rfl) (by ext; rfl)
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem dualQuotEquivDualAnnihilator_apply (W : Submodule R M) (φ : Module.Dual R (M ⧸ W)) (x : M) :
    dualQuotEquivDualAnnihilator W φ x = φ (Quotient.mk x) :=
  rfl


theorem dualCopairing_eq (W : Submodule R M) :
    W.dualCopairing = (dualQuotEquivDualAnnihilator W).symm.toLinearMap :=
  rfl


@[simp]
theorem dualQuotEquivDualAnnihilator_symm_apply_mk (W : Submodule R M) (φ : W.dualAnnihilator)
    (x : M) : (dualQuotEquivDualAnnihilator W).symm φ (Quotient.mk x) = φ x :=
  rfl


theorem finite_dualAnnihilator_iff {W : Submodule R M} [Free R (M ⧸ W)] :
    Module.Finite R W.dualAnnihilator ↔ Module.Finite R (M ⧸ W) :=
  (Finite.equiv_iff W.dualQuotEquivDualAnnihilator.symm).trans (finite_dual_iff R)


open LinearMap in
/-- The pairing between a submodule `W` of a dual module `Dual R M` and the quotient of
`M` by the coannihilator of `W`, which is always nondegenerate. -/
def quotDualCoannihilatorToDual (W : Submodule R (Dual R M)) :
    M ⧸ W.dualCoannihilator →ₗ[R] Dual R W :=
  liftQ _ (flip <| Submodule.subtype _) le_rfl


@[simp]
theorem quotDualCoannihilatorToDual_apply (W : Submodule R (Dual R M)) (m : M) (w : W) :
    W.quotDualCoannihilatorToDual (Quotient.mk m) w = w.1 m := rfl


theorem quotDualCoannihilatorToDual_injective (W : Submodule R (Dual R M)) :
    Function.Injective W.quotDualCoannihilatorToDual :=
  LinearMap.ker_eq_bot.mp (ker_liftQ_eq_bot _ _ _ le_rfl)


theorem flip_quotDualCoannihilatorToDual_injective (W : Submodule R (Dual R M)) :
    Function.Injective W.quotDualCoannihilatorToDual.flip :=
  fun _ _ he ↦ Subtype.ext <| LinearMap.ext fun m ↦ DFunLike.congr_fun he ⟦m⟧


open LinearMap in
theorem quotDualCoannihilatorToDual_nondegenerate (W : Submodule R (Dual R M)) :
    W.quotDualCoannihilatorToDual.Nondegenerate := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    W : Submodule R (Module.Dual R M)
    ⊢ W.quotDualCoannihilatorToDual.Nondegenerate
  -/
  rw [Nondegenerate, separatingLeft_iff_ker_eq_bot, separatingRight_iff_flip_ker_eq_bot]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    W : Submodule R (Module.Dual R M)
    ⊢ And (Eq (LinearMap.ker W.quotDualCoannihilatorToDual) Bot.bot) (Eq (LinearMa …
  -/
  letI : AddCommGroup W := inferInstance
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    W : Submodule R (Module.Dual R M)
    this : AddCommGroup (Subtype fun x => Membership.mem W x) := inferInstance
    ⊢ And (Eq (LinearMap.ker W.quotDualCoannihilatorToDual) Bot.bot) (Eq (LinearMa …
  -/
  simp_rw [ker_eq_bot]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    W : Submodule R (Module.Dual R M)
    this : AddCommGroup (Subtype fun x => Membership.mem W x) := inferInstance
    ⊢ And (Function.Injective ⇑W.quotDualCoannihilatorToDual) (Function.Injective  …
  -/
  exact ⟨W.quotDualCoannihilatorToDual_injective, W.flip_quotDualCoannihilatorToDual_injective⟩
  /-
    🎉 no goals
  -/


theorem range_dualMap_eq_dualAnnihilator_ker_of_surjective (f : M →ₗ[R] M')
    (hf : Function.Surjective f) : LinearMap.range f.dualMap = f.ker.dualAnnihilator :=
  ((f.quotKerEquivOfSurjective hf).dualMap.range_comp _).trans f.ker.range_dualMap_mkQ_eq

-- Note, this can be specialized to the case where `R` is an injective `R`-module, or when
-- `f.coker` is a projective `R`-module.

theorem range_dualMap_eq_dualAnnihilator_ker_of_subtype_range_surjective (f : M →ₗ[R] M')
    (hf : Function.Surjective f.range.subtype.dualMap) :
    LinearMap.range f.dualMap = f.ker.dualAnnihilator := by
  have rr_surj : Function.Surjective f.rangeRestrict := by
    rw [← range_eq_top, range_rangeRestrict]
  /-
    R : Type u_1
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    hf : Function.Surjective ⇑(LinearMap.range f).subtype.dualMap
    rr_surj : Function.Surjective ⇑f.rangeRestrict
    ⊢ Eq (LinearMap.range f.dualMap) (LinearMap.ker f).dualAnnihilator
  -/
  have := range_dualMap_eq_dualAnnihilator_ker_of_surjective f.rangeRestrict rr_surj
  /-
    R : Type u_1
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    hf : Function.Surjective ⇑(LinearMap.range f).subtype.dualMap
    rr_surj : Function.Surjective ⇑f.rangeRestrict
    this : Eq (LinearMap.range f.rangeRestrict.dualMap) (LinearMap.ker f.rangeRest …
    ⊢ Eq (LinearMap.range f.dualMap) (LinearMap.ker f).dualAnnihilator
  -/
  convert this using 1
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629
  · calc
      _ = range ((range f).subtype.comp f.rangeRestrict).dualMap := by simp
      _ = _ := ?_
    /-
      case h.e'_2
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      hf : Function.Surjective ⇑(LinearMap.range f).subtype.dualMap
      rr_surj : Function.Surjective ⇑f.rangeRestrict
      this : Eq (LinearMap.range f.rangeRestrict.dualMap) (LinearMap.ker f.rangeRest …
      ⊢ Eq (LinearMap.range ((LinearMap.range f).subtype.comp f.rangeRestrict).dualM …
    -/
    rw [← dualMap_comp_dualMap, range_comp_of_range_eq_top]
    /-
      case h.e'_2.hf
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      hf : Function.Surjective ⇑(LinearMap.range f).subtype.dualMap
      rr_surj : Function.Surjective ⇑f.rangeRestrict
      this : Eq (LinearMap.range f.rangeRestrict.dualMap) (LinearMap.ker f.rangeRest …
      ⊢ Eq (LinearMap.range (LinearMap.range f).subtype.dualMap) Top.top
    -/
    rwa [range_eq_top]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      hf : Function.Surjective ⇑(LinearMap.range f).subtype.dualMap
      rr_surj : Function.Surjective ⇑f.rangeRestrict
      this : Eq (LinearMap.range f.rangeRestrict.dualMap) (LinearMap.ker f.rangeRest …
      ⊢ Eq (LinearMap.ker f).dualAnnihilator (LinearMap.ker f.rangeRestrict).dualAnn …
    -/
  · apply congr_arg
    /-
      case h.e'_3.h
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      hf : Function.Surjective ⇑(LinearMap.range f).subtype.dualMap
      rr_surj : Function.Surjective ⇑f.rangeRestrict
      this : Eq (LinearMap.range f.rangeRestrict.dualMap) (LinearMap.ker f.rangeRest …
      ⊢ Eq (LinearMap.ker f) (LinearMap.ker f.rangeRestrict)
    -/
    exact (ker_rangeRestrict f).symm
    /-
      🎉 no goals
    -/


theorem ker_dualMap_eq_dualCoannihilator_range (f : M →ₗ[R] M') :
    LinearMap.ker f.dualMap = (Dual.eval R M' ∘ₗ f).range.dualCoannihilator := by
  /-
    R : Type u_1
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    ⊢ Eq (LinearMap.ker f.dualMap) (LinearMap.range ((Module.Dual.eval R M').comp  …
  -/
  ext x; simp [LinearMap.ext_iff (f := dualMap f x)]
         /-
           🎉 no goals
         -/


@[simp]
lemma dualCoannihilator_range_eq_ker_flip (B : M →ₗ[R] M' →ₗ[R] R) :
    (range B).dualCoannihilator = LinearMap.ker B.flip := by
  /-
    R : Type u_1
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M' R)
    ⊢ Eq (LinearMap.range B).dualCoannihilator (LinearMap.ker B.flip)
  -/
  ext x; simp [LinearMap.ext_iff (f := B.flip x)]
         /-
           🎉 no goals
         -/


lemma range_eq_top_of_ne_zero :
    LinearMap.range f = ⊤ := by
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    f : Module.Dual K V₁
    hf : Ne f 0
    ⊢ Eq (LinearMap.range f) Top.top
  -/
  obtain ⟨v, hv⟩ : ∃ v, f v ≠ 0 := by contrapose! hf; ext v; simpa using hf v
  /-
    case intro
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    f : Module.Dual K V₁
    hf : Ne f 0
    v : V₁
    hv : Ne (f v) 0
    ⊢ Eq (LinearMap.range f) Top.top
  -/
  rw [eq_top_iff]
  /-
    case intro
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    f : Module.Dual K V₁
    hf : Ne f 0
    v : V₁
    hv : Ne (f v) 0
    ⊢ LE.le Top.top (LinearMap.range f)
  -/
  exact fun x _ ↦ ⟨x • (f v)⁻¹ • v, by simp [inv_mul_cancel₀ hv]⟩
  /-
    🎉 no goals
  -/


lemma finrank_ker_add_one_of_ne_zero :
    finrank K (LinearMap.ker f) + 1 = finrank K V₁ := by
  suffices finrank K (LinearMap.range f) = 1 by
    rw [← (LinearMap.ker f).finrank_quotient_add_finrank, add_comm, add_left_inj,
    f.quotKerEquivRange.finrank_eq, this]
  /-
    K : Type uK
    inst✝³ : Field K
    V₁ : Type uV₁
    inst✝² : AddCommGroup V₁
    inst✝¹ : Module K V₁
    f : Module.Dual K V₁
    hf : Ne f 0
    inst✝ : FiniteDimensional K V₁
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (LinearMap.range f) x) …
  -/
  rw [range_eq_top_of_ne_zero hf, finrank_top, finrank_self]
  /-
    🎉 no goals
  -/


lemma isCompl_ker_of_disjoint_of_ne_bot {p : Submodule K V₁}
    (hpf : Disjoint (LinearMap.ker f) p) (hp : p ≠ ⊥) :
    IsCompl (LinearMap.ker f) p := by
  /-
    K : Type uK
    inst✝³ : Field K
    V₁ : Type uV₁
    inst✝² : AddCommGroup V₁
    inst✝¹ : Module K V₁
    f : Module.Dual K V₁
    hf : Ne f 0
    inst✝ : FiniteDimensional K V₁
    p : Submodule K V₁
    hpf : Disjoint (LinearMap.ker f) p
    hp : Ne p Bot.bot
    ⊢ IsCompl (LinearMap.ker f) p
  -/
  refine ⟨hpf, codisjoint_iff.mpr <| eq_of_le_of_finrank_le le_top ?_⟩
  have : finrank K ↑(LinearMap.ker f ⊔ p) = finrank K (LinearMap.ker f) + finrank K p := by
    simp [← Submodule.finrank_sup_add_finrank_inf_eq (LinearMap.ker f) p, hpf.eq_bot]
  rwa [finrank_top, this, ← finrank_ker_add_one_of_ne_zero hf, add_le_add_iff_left,
    Submodule.one_le_finrank_iff]


lemma eq_of_ker_eq_of_apply_eq [FiniteDimensional K V₁] {f g : Module.Dual K V₁} (x : V₁)
    (h : LinearMap.ker f = LinearMap.ker g) (h' : f x = g x) (hx : f x ≠ 0) :
    f = g := by
  /-
    K : Type uK
    inst✝³ : Field K
    V₁ : Type uV₁
    inst✝² : AddCommGroup V₁
    inst✝¹ : Module K V₁
    inst✝ : FiniteDimensional K V₁
    f g : Module.Dual K V₁
    x : V₁
    h : Eq (LinearMap.ker f) (LinearMap.ker g)
    h' : Eq (f x) (g x)
    hx : Ne (f x) 0
    ⊢ Eq f g
  -/
  let p := K ∙ x
  /-
    K : Type uK
    inst✝³ : Field K
    V₁ : Type uV₁
    inst✝² : AddCommGroup V₁
    inst✝¹ : Module K V₁
    inst✝ : FiniteDimensional K V₁
    f g : Module.Dual K V₁
    x : V₁
    h : Eq (LinearMap.ker f) (LinearMap.ker g)
    h' : Eq (f x) (g x)
    hx : Ne (f x) 0
    p : Submodule K V₁ := Submodule.span K (Singleton.singleton x)
    ⊢ Eq f g
  -/
  have hp : p ≠ ⊥ := by aesop
  have hpf : Disjoint (LinearMap.ker f) p := by
    rw [disjoint_iff, Submodule.eq_bot_iff]
    rintro y ⟨hfy : f y = 0, hpy : y ∈ p⟩
    obtain ⟨t, rfl⟩ := Submodule.mem_span_singleton.mp hpy
    have ht : t = 0 := by simpa [hx] using hfy
    simp [ht]
  /-
    K : Type uK
    inst✝³ : Field K
    V₁ : Type uV₁
    inst✝² : AddCommGroup V₁
    inst✝¹ : Module K V₁
    inst✝ : FiniteDimensional K V₁
    f g : Module.Dual K V₁
    x : V₁
    h : Eq (LinearMap.ker f) (LinearMap.ker g)
    h' : Eq (f x) (g x)
    hx : Ne (f x) 0
    p : Submodule K V₁ := Submodule.span K (Singleton.singleton x)
    hp : Ne p Bot.bot
    hpf : Disjoint (LinearMap.ker f) p
    ⊢ Eq f g
  -/
  have hf : f ≠ 0 := by aesop
  /-
    K : Type uK
    inst✝³ : Field K
    V₁ : Type uV₁
    inst✝² : AddCommGroup V₁
    inst✝¹ : Module K V₁
    inst✝ : FiniteDimensional K V₁
    f g : Module.Dual K V₁
    x : V₁
    h : Eq (LinearMap.ker f) (LinearMap.ker g)
    h' : Eq (f x) (g x)
    hx : Ne (f x) 0
    p : Submodule K V₁ := Submodule.span K (Singleton.singleton x)
    hp : Ne p Bot.bot
    hpf : Disjoint (LinearMap.ker f) p
    hf : Ne f 0
    ⊢ Eq f g
  -/
  ext v
  obtain ⟨y, hy, z, hz, rfl⟩ : ∃ᵉ (y ∈ LinearMap.ker f) (z ∈ p), y + z = v := by
    have : v ∈ (⊤ : Submodule K V₁) := Submodule.mem_top
    rwa [← (isCompl_ker_of_disjoint_of_ne_bot hf hpf hp).sup_eq_top, Submodule.mem_sup] at this
  /-
    case h.intro.intro.intro.intro
    K : Type uK
    inst✝³ : Field K
    V₁ : Type uV₁
    inst✝² : AddCommGroup V₁
    inst✝¹ : Module K V₁
    inst✝ : FiniteDimensional K V₁
    f g : Module.Dual K V₁
    x : V₁
    h : Eq (LinearMap.ker f) (LinearMap.ker g)
    h' : Eq (f x) (g x)
    hx : Ne (f x) 0
    p : Submodule K V₁ := Submodule.span K (Singleton.singleton x)
    hp : Ne p Bot.bot
    hpf : Disjoint (LinearMap.ker f) p
    hf : Ne f 0
    y : V₁
    hy : Membership.mem (LinearMap.ker f) y
    z : V₁
    hz : Membership.mem p z
    ⊢ Eq (f (HAdd.hAdd y z)) (g (HAdd.hAdd y z))
  -/
  have hy' : g y = 0 := by rwa [← LinearMap.mem_ker, ← h]
  /-
    case h.intro.intro.intro.intro
    K : Type uK
    inst✝³ : Field K
    V₁ : Type uV₁
    inst✝² : AddCommGroup V₁
    inst✝¹ : Module K V₁
    inst✝ : FiniteDimensional K V₁
    f g : Module.Dual K V₁
    x : V₁
    h : Eq (LinearMap.ker f) (LinearMap.ker g)
    h' : Eq (f x) (g x)
    hx : Ne (f x) 0
    p : Submodule K V₁ := Submodule.span K (Singleton.singleton x)
    hp : Ne p Bot.bot
    hpf : Disjoint (LinearMap.ker f) p
    hf : Ne f 0
    y : V₁
    hy : Membership.mem (LinearMap.ker f) y
    z : V₁
    hz : Membership.mem p z
    hy' : Eq (g y) 0
    ⊢ Eq (f (HAdd.hAdd y z)) (g (HAdd.hAdd y z))
  -/
  replace hy : f y = 0 := by rwa [LinearMap.mem_ker] at hy
  /-
    case h.intro.intro.intro.intro
    K : Type uK
    inst✝³ : Field K
    V₁ : Type uV₁
    inst✝² : AddCommGroup V₁
    inst✝¹ : Module K V₁
    inst✝ : FiniteDimensional K V₁
    f g : Module.Dual K V₁
    x : V₁
    h : Eq (LinearMap.ker f) (LinearMap.ker g)
    h' : Eq (f x) (g x)
    hx : Ne (f x) 0
    p : Submodule K V₁ := Submodule.span K (Singleton.singleton x)
    hp : Ne p Bot.bot
    hpf : Disjoint (LinearMap.ker f) p
    hf : Ne f 0
    y z : V₁
    hz : Membership.mem p z
    hy' : Eq (g y) 0
    hy : Eq (f y) 0
    ⊢ Eq (f (HAdd.hAdd y z)) (g (HAdd.hAdd y z))
  -/
  obtain ⟨t, rfl⟩ := Submodule.mem_span_singleton.mp hz
  /-
    case h.intro.intro.intro.intro.intro
    K : Type uK
    inst✝³ : Field K
    V₁ : Type uV₁
    inst✝² : AddCommGroup V₁
    inst✝¹ : Module K V₁
    inst✝ : FiniteDimensional K V₁
    f g : Module.Dual K V₁
    x : V₁
    h : Eq (LinearMap.ker f) (LinearMap.ker g)
    h' : Eq (f x) (g x)
    hx : Ne (f x) 0
    p : Submodule K V₁ := Submodule.span K (Singleton.singleton x)
    hp : Ne p Bot.bot
    hpf : Disjoint (LinearMap.ker f) p
    hf : Ne f 0
    y : V₁
    hy' : Eq (g y) 0
    hy : Eq (f y) 0
    t : K
    hz : Membership.mem p (HSMul.hSMul t x)
    ⊢ Eq (f (HAdd.hAdd y (HSMul.hSMul t x))) (g (HAdd.hAdd y (HSMul.hSMul t x)))
  -/
  simp [h', hy, hy']
  /-
    🎉 no goals
  -/


theorem dualPairing_nondegenerate : (dualPairing K V₁).Nondegenerate :=
  ⟨separatingLeft_iff_ker_eq_bot.mpr ker_id, fun x => (forall_dual_apply_eq_zero_iff K x).mp⟩


theorem dualMap_surjective_of_injective {f : V₁ →ₗ[K] V₂} (hf : Function.Injective f) :
    Function.Surjective f.dualMap := fun φ ↦
  have ⟨f', hf'⟩ := f.exists_leftInverse_of_injective (ker_eq_bot.mpr hf)
  ⟨φ.comp f', ext fun x ↦ congr(φ <| $hf' x)⟩

  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629 LinearMap.range

theorem range_dualMap_eq_dualAnnihilator_ker (f : V₁ →ₗ[K] V₂) :
    LinearMap.range f.dualMap = f.ker.dualAnnihilator :=
  range_dualMap_eq_dualAnnihilator_ker_of_subtype_range_surjective f <|
    dualMap_surjective_of_injective (range f).injective_subtype


/-- For vector spaces, `f.dualMap` is surjective if and only if `f` is injective -/
@[simp]
theorem dualMap_surjective_iff {f : V₁ →ₗ[K] V₂} :
    Function.Surjective f.dualMap ↔ Function.Injective f := by
  rw [← LinearMap.range_eq_top, range_dualMap_eq_dualAnnihilator_ker,
      ← Submodule.dualAnnihilator_bot, Subspace.dualAnnihilator_inj, LinearMap.ker_eq_bot]


theorem dualPairing_eq (W : Subspace K V₁) :
    W.dualPairing = W.quotAnnihilatorEquiv.toLinearMap := by
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W : Subspace K V₁
    ⊢ Eq (Submodule.dualPairing W) ↑W.quotAnnihilatorEquiv
  -/
  ext
  /-
    case h.h.h
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W : Subspace K V₁
    x✝¹ : Module.Dual K V₁
    x✝ : Subtype fun x => Membership.mem W x
    ⊢ Eq ((((Submodule.dualPairing W).comp (Submodule.dualAnnihilator W).mkQ) x✝¹) …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem dualPairing_nondegenerate (W : Subspace K V₁) : W.dualPairing.Nondegenerate := by
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W : Subspace K V₁
    ⊢ (Submodule.dualPairing W).Nondegenerate
  -/
  constructor
    /-
      case left
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      W : Subspace K V₁
      ⊢ (Submodule.dualPairing W).SeparatingLeft
    -/
  · rw [LinearMap.separatingLeft_iff_ker_eq_bot, dualPairing_eq]
    /-
      case left
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      W : Subspace K V₁
      ⊢ Eq (LinearMap.ker ↑W.quotAnnihilatorEquiv) Bot.bot
    -/
    apply LinearEquiv.ker
    /-
      🎉 no goals
    -/
    /-
      case right
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      W : Subspace K V₁
      ⊢ (Submodule.dualPairing W).SeparatingRight
    -/
  · intro x h
    /-
      case right
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      W : Subspace K V₁
      x : Subtype fun x => Membership.mem W x
      h : ∀ (x_1 : HasQuotient.Quotient (Module.Dual K V₁) (Submodule.dualAnnihilato …
      ⊢ Eq x 0
    -/
    rw [← forall_dual_apply_eq_zero_iff K x]
    /-
      case right
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      W : Subspace K V₁
      x : Subtype fun x => Membership.mem W x
      h : ∀ (x_1 : HasQuotient.Quotient (Module.Dual K V₁) (Submodule.dualAnnihilato …
      ⊢ ∀ (φ : Module.Dual K (Subtype fun x => Membership.mem W x)), Eq (φ x) 0
    -/
    intro φ
    simpa only [Submodule.dualPairing_apply, dualLift_of_subtype] using
      h (Submodule.Quotient.mk (W.dualLift φ))


theorem dualCopairing_nondegenerate (W : Subspace K V₁) : W.dualCopairing.Nondegenerate := by
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W : Subspace K V₁
    ⊢ (Submodule.dualCopairing W).Nondegenerate
  -/
  constructor
    /-
      case left
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      W : Subspace K V₁
      ⊢ (Submodule.dualCopairing W).SeparatingLeft
    -/
  · rw [LinearMap.separatingLeft_iff_ker_eq_bot, dualCopairing_eq]
    /-
      case left
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      W : Subspace K V₁
      ⊢ Eq (LinearMap.ker ↑(Submodule.dualQuotEquivDualAnnihilator W).symm) Bot.bot
    -/
    apply LinearEquiv.ker
    /-
      🎉 no goals
    -/
    /-
      case right
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      W : Subspace K V₁
      ⊢ (Submodule.dualCopairing W).SeparatingRight
    -/
  · rintro ⟨x⟩
    /-
      case right.mk
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      W : Subspace K V₁
      y✝ : HasQuotient.Quotient V₁ W
      x : V₁
      ⊢ (∀ (x_1 : Subtype fun x => Membership.mem (Submodule.dualAnnihilator W) x),  …
    -/
    simp only [Quotient.quot_mk_eq_mk, dualCopairing_apply, Quotient.mk_eq_zero]
    /-
      case right.mk
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      W : Subspace K V₁
      y✝ : HasQuotient.Quotient V₁ W
      x : V₁
      ⊢ (∀ (x_1 : Subtype fun x => Membership.mem (Submodule.dualAnnihilator W) x),  …
    -/
    rw [← forall_mem_dualAnnihilator_apply_eq_zero_iff, SetLike.forall]
    /-
      case right.mk
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      W : Subspace K V₁
      y✝ : HasQuotient.Quotient V₁ W
      x : V₁
      ⊢ (∀ (x_1 : Module.Dual K V₁) (h : Membership.mem (Submodule.dualAnnihilator W …
    -/
    exact id
    /-
      🎉 no goals
    -/

-- Argument from https://math.stackexchange.com/a/2423263/172988

theorem dualAnnihilator_inf_eq (W W' : Subspace K V₁) :
    (W ⊓ W').dualAnnihilator = W.dualAnnihilator ⊔ W'.dualAnnihilator := by
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    ⊢ Eq (Submodule.dualAnnihilator (Min.min W W')) (Max.max (Submodule.dualAnnihi …
  -/
  refine le_antisymm ?_ (sup_dualAnnihilator_le_inf W W')
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    ⊢ LE.le (Submodule.dualAnnihilator (Min.min W W')) (Max.max (Submodule.dualAnn …
  -/
  let F : V₁ →ₗ[K] (V₁ ⧸ W) × V₁ ⧸ W' := (Submodule.mkQ W).prod (Submodule.mkQ W')
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629 LinearMap.ker
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    F : LinearMap (RingHom.id K) V₁ (Prod (HasQuotient.Quotient V₁ W) (HasQuotient …
    ⊢ LE.le (Submodule.dualAnnihilator (Min.min W W')) (Max.max (Submodule.dualAnn …
  -/
  have : LinearMap.ker F = W ⊓ W' := by simp only [F, LinearMap.ker_prod, ker_mkQ]
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    F : LinearMap (RingHom.id K) V₁ (Prod (HasQuotient.Quotient V₁ W) (HasQuotient …
    this : Eq (LinearMap.ker F) (Min.min W W')
    ⊢ LE.le (Submodule.dualAnnihilator (Min.min W W')) (Max.max (Submodule.dualAnn …
  -/
  rw [← this, ← LinearMap.range_dualMap_eq_dualAnnihilator_ker]
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    F : LinearMap (RingHom.id K) V₁ (Prod (HasQuotient.Quotient V₁ W) (HasQuotient …
    this : Eq (LinearMap.ker F) (Min.min W W')
    ⊢ LE.le (LinearMap.range F.dualMap) (Max.max (Submodule.dualAnnihilator W) (Su …
  -/
  intro φ
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    F : LinearMap (RingHom.id K) V₁ (Prod (HasQuotient.Quotient V₁ W) (HasQuotient …
    this : Eq (LinearMap.ker F) (Min.min W W')
    φ : Module.Dual K V₁
    ⊢ Membership.mem (LinearMap.range F.dualMap) φ → Membership.mem (Max.max (Subm …
  -/
  rw [LinearMap.mem_range]
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    F : LinearMap (RingHom.id K) V₁ (Prod (HasQuotient.Quotient V₁ W) (HasQuotient …
    this : Eq (LinearMap.ker F) (Min.min W W')
    φ : Module.Dual K V₁
    ⊢ (Exists fun y => Eq (F.dualMap y) φ) → Membership.mem (Max.max (Submodule.du …
  -/
  rintro ⟨x, rfl⟩
  /-
    case intro
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    F : LinearMap (RingHom.id K) V₁ (Prod (HasQuotient.Quotient V₁ W) (HasQuotient …
    this : Eq (LinearMap.ker F) (Min.min W W')
    x : Module.Dual K (Prod (HasQuotient.Quotient V₁ W) (HasQuotient.Quotient V₁ W …
    ⊢ Membership.mem (Max.max (Submodule.dualAnnihilator W) (Submodule.dualAnnihil …
  -/
  rw [Submodule.mem_sup]
  /-
    case intro
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    F : LinearMap (RingHom.id K) V₁ (Prod (HasQuotient.Quotient V₁ W) (HasQuotient …
    this : Eq (LinearMap.ker F) (Min.min W W')
    x : Module.Dual K (Prod (HasQuotient.Quotient V₁ W) (HasQuotient.Quotient V₁ W …
    ⊢ Exists fun y => And (Membership.mem (Submodule.dualAnnihilator W) y) (Exists …
  -/
  obtain ⟨⟨a, b⟩, rfl⟩ := (dualProdDualEquivDual K (V₁ ⧸ W) (V₁ ⧸ W')).surjective x
  /-
    case intro.intro.mk
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    F : LinearMap (RingHom.id K) V₁ (Prod (HasQuotient.Quotient V₁ W) (HasQuotient …
    this : Eq (LinearMap.ker F) (Min.min W W')
    a : Module.Dual K (HasQuotient.Quotient V₁ W)
    b : Module.Dual K (HasQuotient.Quotient V₁ W')
    ⊢ Exists fun y => And (Membership.mem (Submodule.dualAnnihilator W) y) (Exists …
  -/
  obtain ⟨a', rfl⟩ := (dualQuotEquivDualAnnihilator W).symm.surjective a
  /-
    case intro.intro.mk.intro
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    F : LinearMap (RingHom.id K) V₁ (Prod (HasQuotient.Quotient V₁ W) (HasQuotient …
    this : Eq (LinearMap.ker F) (Min.min W W')
    b : Module.Dual K (HasQuotient.Quotient V₁ W')
    a' : Subtype fun x => Membership.mem (Submodule.dualAnnihilator W) x
    ⊢ Exists fun y => And (Membership.mem (Submodule.dualAnnihilator W) y) (Exists …
  -/
  obtain ⟨b', rfl⟩ := (dualQuotEquivDualAnnihilator W').symm.surjective b
  /-
    case intro.intro.mk.intro.intro
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    F : LinearMap (RingHom.id K) V₁ (Prod (HasQuotient.Quotient V₁ W) (HasQuotient …
    this : Eq (LinearMap.ker F) (Min.min W W')
    a' : Subtype fun x => Membership.mem (Submodule.dualAnnihilator W) x
    b' : Subtype fun x => Membership.mem (Submodule.dualAnnihilator W') x
    ⊢ Exists fun y => And (Membership.mem (Submodule.dualAnnihilator W) y) (Exists …
  -/
  use a', a'.property, b', b'.property
  /-
    case right
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    F : LinearMap (RingHom.id K) V₁ (Prod (HasQuotient.Quotient V₁ W) (HasQuotient …
    this : Eq (LinearMap.ker F) (Min.min W W')
    a' : Subtype fun x => Membership.mem (Submodule.dualAnnihilator W) x
    b' : Subtype fun x => Membership.mem (Submodule.dualAnnihilator W') x
    ⊢ Eq (HAdd.hAdd ↑a' ↑b') (F.dualMap ((Module.dualProdDualEquivDual K (HasQuoti …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- This is also true if `V₁` is finite dimensional since one can restrict `ι` to some subtype
-- for which the infi and supr are the same.
-- The obstruction to the `dualAnnihilator_inf_eq` argument carrying through is that we need
-- for `Module.Dual R (Π (i : ι), V ⧸ W i) ≃ₗ[K] Π (i : ι), Module.Dual R (V ⧸ W i)`, which is not
-- true for infinite `ι`. One would need to add additional hypothesis on `W` (for example, it might
-- be true when the family is inf-closed).
-- TODO: generalize to `Sort`

theorem dualAnnihilator_iInf_eq {ι : Type*} [Finite ι] (W : ι → Subspace K V₁) :
    (⨅ i : ι, W i).dualAnnihilator = ⨆ i : ι, (W i).dualAnnihilator := by
  /-
    K : Type uK
    inst✝³ : Field K
    V₁ : Type uV₁
    inst✝² : AddCommGroup V₁
    inst✝¹ : Module K V₁
    ι : Type u_1
    inst✝ : Finite ι
    W : ι → Subspace K V₁
    ⊢ Eq (Submodule.dualAnnihilator (iInf fun i => W i)) (iSup fun i => Submodule. …
  -/
  revert ι
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    ⊢ ∀ {ι : Type u_1} [inst : Finite ι] (W : ι → Subspace K V₁), Eq (Submodule.du …
  -/
  apply Finite.induction_empty_option
    /-
      case of_equiv
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      ⊢ ∀ {α β : Type u_1}, Equiv α β → (∀ (W : α → Subspace K V₁), Eq (Submodule.du …
    -/
  · intro α β h hyp W
    /-
      case of_equiv
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      α β : Type u_1
      h : Equiv α β
      hyp : ∀ (W : α → Subspace K V₁), Eq (Submodule.dualAnnihilator (iInf fun i =>  …
      W : β → Subspace K V₁
      ⊢ Eq (Submodule.dualAnnihilator (iInf fun i => W i)) (iSup fun i => Submodule. …
    -/
    rw [← h.iInf_comp, hyp _, ← h.iSup_comp]
    /-
      🎉 no goals
    -/
    /-
      case h_empty
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      ⊢ ∀ (W : PEmpty.{u_1 + 1} → Subspace K V₁), Eq (Submodule.dualAnnihilator (iIn …
    -/
  · intro W
    /-
      case h_empty
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      W : PEmpty.{u_1 + 1} → Subspace K V₁
      ⊢ Eq (Submodule.dualAnnihilator (iInf fun i => W i)) (iSup fun i => Submodule. …
    -/
    rw [iSup_of_empty', iInf_of_isEmpty, sInf_empty, sSup_empty, dualAnnihilator_top]
    /-
      🎉 no goals
    -/
    /-
      case h_option
      K : Type uK
      inst✝² : Field K
      V₁ : Type uV₁
      inst✝¹ : AddCommGroup V₁
      inst✝ : Module K V₁
      ⊢ ∀ {α : Type u_1} [inst : Fintype α], (∀ (W : α → Subspace K V₁), Eq (Submodu …
    -/
  · intro α _ h W
    /-
      case h_option
      K : Type uK
      inst✝³ : Field K
      V₁ : Type uV₁
      inst✝² : AddCommGroup V₁
      inst✝¹ : Module K V₁
      α : Type u_1
      inst✝ : Fintype α
      h : ∀ (W : α → Subspace K V₁), Eq (Submodule.dualAnnihilator (iInf fun i => W  …
      W : Option α → Subspace K V₁
      ⊢ Eq (Submodule.dualAnnihilator (iInf fun i => W i)) (iSup fun i => Submodule. …
    -/
    rw [iInf_option, iSup_option, dualAnnihilator_inf_eq, h]
    /-
      🎉 no goals
    -/


/-- For vector spaces, dual annihilators carry direct sum decompositions
to direct sum decompositions. -/
theorem isCompl_dualAnnihilator {W W' : Subspace K V₁} (h : IsCompl W W') :
    IsCompl W.dualAnnihilator W'.dualAnnihilator := by
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    h : IsCompl W W'
    ⊢ IsCompl (Submodule.dualAnnihilator W) (Submodule.dualAnnihilator W')
  -/
  rw [isCompl_iff, disjoint_iff, codisjoint_iff] at h ⊢
  rw [← dualAnnihilator_inf_eq, ← dualAnnihilator_sup_eq, h.1, h.2, dualAnnihilator_top,
    dualAnnihilator_bot]
  /-
    K : Type uK
    inst✝² : Field K
    V₁ : Type uV₁
    inst✝¹ : AddCommGroup V₁
    inst✝ : Module K V₁
    W W' : Subspace K V₁
    h : And (Eq (Min.min W W') Bot.bot) (Eq (Max.max W W') Top.top)
    ⊢ And (Eq Bot.bot Bot.bot) (Eq Top.top Top.top)
  -/
  exact ⟨rfl, rfl⟩
  /-
    🎉 no goals
  -/


/-- For finite-dimensional vector spaces, one can distribute duals over quotients by identifying
`W.dualLift.range` with `W`. Note that this depends on a choice of splitting of `V₁`. -/
def dualQuotDistrib [FiniteDimensional K V₁] (W : Subspace K V₁) :
    Module.Dual K (V₁ ⧸ W) ≃ₗ[K] Module.Dual K V₁ ⧸ LinearMap.range W.dualLift :=
  W.dualQuotEquivDualAnnihilator.trans W.quotDualEquivAnnihilator.symm


@[simp]
theorem finrank_range_dualMap_eq_finrank_range (f : V₁ →ₗ[K] V₂) :
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation https://github.com/leanprover/lean4/issues/1629
    finrank K (LinearMap.range f.dualMap) = finrank K (LinearMap.range f) := by
  rw [congr_arg dualMap (show f = (range f).subtype.comp f.rangeRestrict by rfl),
    ← dualMap_comp_dualMap, range_comp,
    range_eq_top.mpr (dualMap_surjective_of_injective (range f).injective_subtype),
    Submodule.map_top, finrank_range_of_inj, Subspace.dual_finrank_eq]
  /-
    K : Type uK
    inst✝⁴ : Field K
    V₁ : Type uV₁
    V₂ : Type uV₂
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    f : LinearMap (RingHom.id K) V₁ V₂
    ⊢ Function.Injective ⇑f.rangeRestrict.dualMap
  -/
  exact dualMap_injective_of_surjective (range_eq_top.mp f.range_rangeRestrict)
  /-
    🎉 no goals
  -/


/-- `f.dualMap` is injective if and only if `f` is surjective -/
@[simp]
theorem dualMap_injective_iff {f : V₁ →ₗ[K] V₂} :
    Function.Injective f.dualMap ↔ Function.Surjective f := by
  /-
    K : Type uK
    inst✝⁴ : Field K
    V₁ : Type uV₁
    V₂ : Type uV₂
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    f : LinearMap (RingHom.id K) V₁ V₂
    ⊢ Iff (Function.Injective ⇑f.dualMap) (Function.Surjective ⇑f)
  -/
  refine ⟨Function.mtr fun not_surj inj ↦ ?_, dualMap_injective_of_surjective⟩
  /-
    K : Type uK
    inst✝⁴ : Field K
    V₁ : Type uV₁
    V₂ : Type uV₂
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    f : LinearMap (RingHom.id K) V₁ V₂
    not_surj : Not (Function.Surjective ⇑f)
    inj : Function.Injective ⇑f.dualMap
    ⊢ False
  -/
  rw [← range_eq_top, ← Ne, ← lt_top_iff_ne_top] at not_surj
  /-
    K : Type uK
    inst✝⁴ : Field K
    V₁ : Type uV₁
    V₂ : Type uV₂
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    f : LinearMap (RingHom.id K) V₁ V₂
    not_surj : LT.lt (LinearMap.range f) Top.top
    inj : Function.Injective ⇑f.dualMap
    ⊢ False
  -/
  obtain ⟨φ, φ0, range_le_ker⟩ := (range f).exists_le_ker_of_lt_top not_surj
  /-
    case intro.intro
    K : Type uK
    inst✝⁴ : Field K
    V₁ : Type uV₁
    V₂ : Type uV₂
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    f : LinearMap (RingHom.id K) V₁ V₂
    not_surj : LT.lt (LinearMap.range f) Top.top
    inj : Function.Injective ⇑f.dualMap
    φ : LinearMap (RingHom.id K) V₂ K
    φ0 : Ne φ 0
    range_le_ker : LE.le (LinearMap.range f) (LinearMap.ker φ)
    ⊢ False
  -/
  exact φ0 (inj <| ext fun x ↦ range_le_ker ⟨x, rfl⟩)
  /-
    🎉 no goals
  -/


/-- `f.dualMap` is bijective if and only if `f` is -/
@[simp]
theorem dualMap_bijective_iff {f : V₁ →ₗ[K] V₂} :
    Function.Bijective f.dualMap ↔ Function.Bijective f := by
  /-
    K : Type uK
    inst✝⁴ : Field K
    V₁ : Type uV₁
    V₂ : Type uV₂
    inst✝³ : AddCommGroup V₁
    inst✝² : Module K V₁
    inst✝¹ : AddCommGroup V₂
    inst✝ : Module K V₂
    f : LinearMap (RingHom.id K) V₁ V₂
    ⊢ Iff (Function.Bijective ⇑f.dualMap) (Function.Bijective ⇑f)
  -/
  simp_rw [Function.Bijective, dualMap_surjective_iff, dualMap_injective_iff, and_comm]
  /-
    🎉 no goals
  -/


@[simp]
lemma dualAnnihilator_ker_eq_range_flip [IsReflexive K V₂] :
    (ker B).dualAnnihilator = range B.flip := by
  /-
    K : Type uK
    inst✝⁵ : Field K
    V₁ : Type uV₁
    V₂ : Type uV₂
    inst✝⁴ : AddCommGroup V₁
    inst✝³ : Module K V₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module K V₂
    B : LinearMap (RingHom.id K) V₁ (LinearMap (RingHom.id K) V₂ K)
    inst✝ : Module.IsReflexive K V₂
    ⊢ Eq (LinearMap.ker B).dualAnnihilator (LinearMap.range B.flip)
  -/
  change _ = range (B.dualMap.comp (Module.evalEquiv K V₂).toLinearMap)
  /-
    K : Type uK
    inst✝⁵ : Field K
    V₁ : Type uV₁
    V₂ : Type uV₂
    inst✝⁴ : AddCommGroup V₁
    inst✝³ : Module K V₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module K V₂
    B : LinearMap (RingHom.id K) V₁ (LinearMap (RingHom.id K) V₂ K)
    inst✝ : Module.IsReflexive K V₂
    ⊢ Eq (LinearMap.ker B).dualAnnihilator (LinearMap.range (B.dualMap.comp ↑(Modu …
  -/
  rw [← range_dualMap_eq_dualAnnihilator_ker, range_comp_of_range_eq_top _ (LinearEquiv.range _)]
  /-
    🎉 no goals
  -/


theorem flip_injective_iff₁ [FiniteDimensional K V₁] : Injective B.flip ↔ Surjective B := by
  /-
    K : Type uK
    inst✝⁵ : Field K
    V₁ : Type uV₁
    V₂ : Type uV₂
    inst✝⁴ : AddCommGroup V₁
    inst✝³ : Module K V₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module K V₂
    B : LinearMap (RingHom.id K) V₁ (LinearMap (RingHom.id K) V₂ K)
    inst✝ : FiniteDimensional K V₁
    ⊢ Iff (Function.Injective ⇑B.flip) (Function.Surjective ⇑B)
  -/
  rw [← dualMap_surjective_iff, ← (evalEquiv K V₁).toEquiv.surjective_comp]; rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem flip_injective_iff₂ [FiniteDimensional K V₂] : Injective B.flip ↔ Surjective B := by
  /-
    K : Type uK
    inst✝⁵ : Field K
    V₁ : Type uV₁
    V₂ : Type uV₂
    inst✝⁴ : AddCommGroup V₁
    inst✝³ : Module K V₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module K V₂
    B : LinearMap (RingHom.id K) V₁ (LinearMap (RingHom.id K) V₂ K)
    inst✝ : FiniteDimensional K V₂
    ⊢ Iff (Function.Injective ⇑B.flip) (Function.Surjective ⇑B)
  -/
  rw [← dualMap_injective_iff]; exact (evalEquiv K V₂).toEquiv.injective_comp B.dualMap
                                /-
                                  🎉 no goals
                                -/


theorem flip_surjective_iff₁ [FiniteDimensional K V₁] : Surjective B.flip ↔ Injective B :=
  flip_injective_iff₂.symm


theorem flip_surjective_iff₂ [FiniteDimensional K V₂] : Surjective B.flip ↔ Injective B :=
  flip_injective_iff₁.symm


theorem flip_bijective_iff₁ [FiniteDimensional K V₁] : Bijective B.flip ↔ Bijective B := by
  /-
    K : Type uK
    inst✝⁵ : Field K
    V₁ : Type uV₁
    V₂ : Type uV₂
    inst✝⁴ : AddCommGroup V₁
    inst✝³ : Module K V₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module K V₂
    B : LinearMap (RingHom.id K) V₁ (LinearMap (RingHom.id K) V₂ K)
    inst✝ : FiniteDimensional K V₁
    ⊢ Iff (Function.Bijective ⇑B.flip) (Function.Bijective ⇑B)
  -/
  simp_rw [Bijective, flip_injective_iff₁, flip_surjective_iff₁, and_comm]
  /-
    🎉 no goals
  -/


theorem flip_bijective_iff₂ [FiniteDimensional K V₂] : Bijective B.flip ↔ Bijective B :=
  flip_bijective_iff₁.symm


theorem quotDualCoannihilatorToDual_bijective (W : Subspace K (Dual K V)) [FiniteDimensional K W] :
    Function.Bijective W.quotDualCoannihilatorToDual :=
  ⟨W.quotDualCoannihilatorToDual_injective, letI : AddCommGroup W := inferInstance
    flip_injective_iff₂.mp W.flip_quotDualCoannihilatorToDual_injective⟩


theorem flip_quotDualCoannihilatorToDual_bijective (W : Subspace K (Dual K V))
    [FiniteDimensional K W] : Function.Bijective W.quotDualCoannihilatorToDual.flip :=
  letI : AddCommGroup W := inferInstance
  flip_bijective_iff₂.mpr W.quotDualCoannihilatorToDual_bijective


theorem dualCoannihilator_dualAnnihilator_eq {W : Subspace K (Dual K V)} [FiniteDimensional K W] :
    W.dualCoannihilator.dualAnnihilator = W :=
  let e := (LinearEquiv.ofBijective _ W.flip_quotDualCoannihilatorToDual_bijective).trans
    (Submodule.dualQuotEquivDualAnnihilator _)
  letI : AddCommGroup W := inferInstance
  haveI : FiniteDimensional K W.dualCoannihilator.dualAnnihilator := LinearEquiv.finiteDimensional e
  (eq_of_le_of_finrank_eq W.le_dualCoannihilator_dualAnnihilator e.finrank_eq).symm


theorem finiteDimensional_quot_dualCoannihilator_iff {W : Submodule K (Dual K V)} :
    FiniteDimensional K (V ⧸ W.dualCoannihilator) ↔ FiniteDimensional K W :=
  ⟨fun _ ↦ FiniteDimensional.of_injective _ W.flip_quotDualCoannihilatorToDual_injective,
    fun _ ↦ by
      #adaptation_note
      /--
      After https://github.com/leanprover/lean4/pull/4119
      the `Free K W` instance isn't found unless we use `set_option maxSynthPendingDepth 2`, or add
      explicit instances:
      ```
      have := Free.of_divisionRing K ↥W
      have := Basis.dual_finite (R := K) (M := W)
      ```
      -/
      set_option maxSynthPendingDepth 2 in
      exact FiniteDimensional.of_injective _ W.quotDualCoannihilatorToDual_injective⟩


open OrderDual in
/-- For any vector space, `dualAnnihilator` and `dualCoannihilator` gives an antitone order
  isomorphism between the finite-codimensional subspaces in the vector space and the
  finite-dimensional subspaces in its dual. -/
def orderIsoFiniteCodimDim :
    {W : Subspace K V // FiniteDimensional K (V ⧸ W)} ≃o
    {W : Subspace K (Dual K V) // FiniteDimensional K W}ᵒᵈ where
  toFun W := toDual ⟨W.1.dualAnnihilator, Submodule.finite_dualAnnihilator_iff.mpr W.2⟩
  invFun W := ⟨(ofDual W).1.dualCoannihilator,
    finiteDimensional_quot_dualCoannihilator_iff.mpr (ofDual W).2⟩
  left_inv _ := Subtype.ext dualAnnihilator_dualCoannihilator_eq
  right_inv W := have := (ofDual W).2; Subtype.ext dualCoannihilator_dualAnnihilator_eq
  map_rel_iff' := dualAnnihilator_le_dualAnnihilator_iff


open OrderDual in
/-- For any finite-dimensional vector space, `dualAnnihilator` and `dualCoannihilator` give
  an antitone order isomorphism between the subspaces in the vector space and the subspaces
  in its dual. -/
def orderIsoFiniteDimensional [FiniteDimensional K V] :
    Subspace K V ≃o (Subspace K (Dual K V))ᵒᵈ where
  toFun W := toDual W.dualAnnihilator
  invFun W := (ofDual W).dualCoannihilator
  left_inv _ := dualAnnihilator_dualCoannihilator_eq
  right_inv _ := dualCoannihilator_dualAnnihilator_eq
  map_rel_iff' := dualAnnihilator_le_dualAnnihilator_iff


open Submodule in
theorem dualAnnihilator_dualAnnihilator_eq_map (W : Subspace K V) [FiniteDimensional K W] :
    W.dualAnnihilator.dualAnnihilator = W.map (Dual.eval K V) := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    W : Subspace K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem W x)
    ⊢ Eq (Submodule.dualAnnihilator W).dualAnnihilator (Submodule.map (Module.Dual …
  -/
  let e1 := (Free.chooseBasis K W).toDualEquiv ≪≫ₗ W.quotAnnihilatorEquiv.symm
  /-
    K : Type u_1
    V : Type u_2
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    W : Subspace K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem W x)
    e1 : LinearEquiv (RingHom.id K) (Subtype fun x => Membership.mem W x) (HasQuot …
    ⊢ Eq (Submodule.dualAnnihilator W).dualAnnihilator (Submodule.map (Module.Dual …
  -/
  haveI := e1.finiteDimensional
  /-
    K : Type u_1
    V : Type u_2
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    W : Subspace K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem W x)
    e1 : LinearEquiv (RingHom.id K) (Subtype fun x => Membership.mem W x) (HasQuot …
    this : FiniteDimensional K (HasQuotient.Quotient (Module.Dual K V) (Submodule. …
    ⊢ Eq (Submodule.dualAnnihilator W).dualAnnihilator (Submodule.map (Module.Dual …
  -/
  let e2 := (Free.chooseBasis K _).toDualEquiv ≪≫ₗ W.dualAnnihilator.dualQuotEquivDualAnnihilator
  /-
    K : Type u_1
    V : Type u_2
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    W : Subspace K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem W x)
    e1 : LinearEquiv (RingHom.id K) (Subtype fun x => Membership.mem W x) (HasQuot …
    this : FiniteDimensional K (HasQuotient.Quotient (Module.Dual K V) (Submodule. …
    e2 : LinearEquiv (RingHom.id K) (HasQuotient.Quotient (Module.Dual K V) (Submo …
    ⊢ Eq (Submodule.dualAnnihilator W).dualAnnihilator (Submodule.map (Module.Dual …
  -/
  haveI := LinearEquiv.finiteDimensional (V₂ := W.dualAnnihilator.dualAnnihilator) e2
  /-
    K : Type u_1
    V : Type u_2
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    W : Subspace K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem W x)
    e1 : LinearEquiv (RingHom.id K) (Subtype fun x => Membership.mem W x) (HasQuot …
    this✝ : FiniteDimensional K (HasQuotient.Quotient (Module.Dual K V) (Submodule …
    e2 : LinearEquiv (RingHom.id K) (HasQuotient.Quotient (Module.Dual K V) (Submo …
    this : FiniteDimensional K (Subtype fun x => Membership.mem (Submodule.dualAnn …
    ⊢ Eq (Submodule.dualAnnihilator W).dualAnnihilator (Submodule.map (Module.Dual …
  -/
  rw [eq_of_le_of_finrank_eq (map_le_dualAnnihilator_dualAnnihilator W)]
  /-
    K : Type u_1
    V : Type u_2
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    W : Subspace K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem W x)
    e1 : LinearEquiv (RingHom.id K) (Subtype fun x => Membership.mem W x) (HasQuot …
    this✝ : FiniteDimensional K (HasQuotient.Quotient (Module.Dual K V) (Submodule …
    e2 : LinearEquiv (RingHom.id K) (HasQuotient.Quotient (Module.Dual K V) (Submo …
    this : FiniteDimensional K (Subtype fun x => Membership.mem (Submodule.dualAnn …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (Submodule.map (Module …
  -/
  rw [← (equivMapOfInjective _ (eval_apply_injective K (V := V)) W).finrank_eq, e1.finrank_eq]
  /-
    K : Type u_1
    V : Type u_2
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    W : Subspace K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem W x)
    e1 : LinearEquiv (RingHom.id K) (Subtype fun x => Membership.mem W x) (HasQuot …
    this✝ : FiniteDimensional K (HasQuotient.Quotient (Module.Dual K V) (Submodule …
    e2 : LinearEquiv (RingHom.id K) (HasQuotient.Quotient (Module.Dual K V) (Submo …
    this : FiniteDimensional K (Subtype fun x => Membership.mem (Submodule.dualAnn …
    ⊢ Eq (Module.finrank K (HasQuotient.Quotient (Module.Dual K V) (Submodule.dual …
  -/
  exact e2.finrank_eq
  /-
    🎉 no goals
  -/


theorem map_dualCoannihilator (W : Subspace K (Dual K V)) [FiniteDimensional K V] :
    W.dualCoannihilator.map (Dual.eval K V) = W.dualAnnihilator := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    W : Subspace K (Module.Dual K V)
    inst✝ : FiniteDimensional K V
    ⊢ Eq (Submodule.map (Module.Dual.eval K V) (Submodule.dualCoannihilator W)) (S …
  -/
  rw [← dualAnnihilator_dualAnnihilator_eq_map, dualCoannihilator_dualAnnihilator_eq]
  /-
    🎉 no goals
  -/


/-- The canonical linear map from `Dual M ⊗ Dual N` to `Dual (M ⊗ N)`,
sending `f ⊗ g` to the composition of `TensorProduct.map f g` with
the natural isomorphism `R ⊗ R ≃ R`.
-/
def dualDistrib : Dual R M ⊗[R] Dual R N →ₗ[R] Dual R (M ⊗[R] N) :=
  compRight ↑(TensorProduct.lid R R) ∘ₗ homTensorHomMap R M N R R


@[simp]
theorem dualDistrib_apply (f : Dual R M) (g : Dual R N) (m : M) (n : N) :
    dualDistrib R M N (f ⊗ₜ g) (m ⊗ₜ n) = f m * g n :=
  rfl


/-- Heterobasic version of `TensorProduct.dualDistrib` -/
def dualDistrib : Dual A M ⊗[R] Dual R N →ₗ[A] Dual A (M ⊗[R] N) :=
  compRight (Algebra.TensorProduct.rid R A A).toLinearMap ∘ₗ homTensorHomMap R A A M N A R


@[simp]
theorem dualDistrib_apply (f : Dual A M) (g : Dual R N) (m : M) (n : N) :
    dualDistrib R A M N (f ⊗ₜ g) (m ⊗ₜ n) = g n • f m :=
  rfl


/-- An inverse to `TensorProduct.dualDistrib` given bases.
-/
noncomputable def dualDistribInvOfBasis (b : Basis ι R M) (c : Basis κ R N) :
    Dual R (M ⊗[R] N) →ₗ[R] Dual R M ⊗[R] Dual R N :=
  -- Porting note: ∑ (i) (j) does not seem to work; applyₗ needs a little help to unify
  ∑ i, ∑ j,
    (ringLmapEquivSelf R ℕ _).symm (b.dualBasis i ⊗ₜ c.dualBasis j) ∘ₗ
      (applyₗ (R := R) (c j)) ∘ₗ (applyₗ (R := R) (b i)) ∘ₗ lcurry R M N R


@[simp]
theorem dualDistribInvOfBasis_apply (b : Basis ι R M) (c : Basis κ R N) (f : Dual R (M ⊗[R] N)) :
    dualDistribInvOfBasis b c f = ∑ i, ∑ j, f (b i ⊗ₜ c j) • b.dualBasis i ⊗ₜ c.dualBasis j := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    b : Basis ι R M
    c : Basis κ R N
    f : Module.Dual R (TensorProduct R M N)
    ⊢ Eq ((TensorProduct.dualDistribInvOfBasis b c) f) (Finset.univ.sum fun i => F …
  -/
  simp [dualDistribInvOfBasis]
  /-
    🎉 no goals
  -/

-- Porting note: introduced to help with timeout in dualDistribEquivOfBasis

theorem dualDistrib_dualDistribInvOfBasis_left_inverse (b : Basis ι R M) (c : Basis κ R N) :
    comp (dualDistrib R M N) (dualDistribInvOfBasis b c) = LinearMap.id := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    b : Basis ι R M
    c : Basis κ R N
    ⊢ Eq ((TensorProduct.dualDistrib R M N).comp (TensorProduct.dualDistribInvOfBa …
  -/
  apply (b.tensorProduct c).dualBasis.ext
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    b : Basis ι R M
    c : Basis κ R N
    ⊢ ∀ (i : Prod ι κ), Eq (((TensorProduct.dualDistrib R M N).comp (TensorProduct …
  -/
  rintro ⟨i, j⟩
  /-
    case mk
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    b : Basis ι R M
    c : Basis κ R N
    i : ι
    j : κ
    ⊢ Eq (((TensorProduct.dualDistrib R M N).comp (TensorProduct.dualDistribInvOfB …
  -/
  apply (b.tensorProduct c).ext
  /-
    case mk
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    b : Basis ι R M
    c : Basis κ R N
    i : ι
    j : κ
    ⊢ ∀ (i_1 : Prod ι κ), Eq ((((TensorProduct.dualDistrib R M N).comp (TensorProd …
  -/
  rintro ⟨i', j'⟩
  simp only [dualDistrib, Basis.coe_dualBasis, coe_comp, Function.comp_apply,
    dualDistribInvOfBasis_apply, Basis.coord_apply, Basis.tensorProduct_repr_tmul_apply,
    Basis.repr_self, ne_eq, _root_.map_sum, map_smul, homTensorHomMap_apply, compRight_apply,
    Basis.tensorProduct_apply, coeFn_sum, Finset.sum_apply, smul_apply, LinearEquiv.coe_coe,
    map_tmul, lid_tmul, smul_eq_mul, id_coe, id_eq]
  /-
    case mk.mk
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    b : Basis ι R M
    c : Basis κ R N
    i : ι
    j : κ
    i' : ι
    j' : κ
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun x_1 => HMul.hMul (HMul.hMul …
  -/
  rw [Finset.sum_eq_single i, Finset.sum_eq_single j]
    /-
      case mk.mk
      R : Type u_1
      M : Type u_3
      N : Type u_4
      ι : Type u_5
      κ : Type u_6
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : DecidableEq κ
      inst✝⁶ : Fintype ι
      inst✝⁵ : Fintype κ
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      b : Basis ι R M
      c : Basis κ R N
      i : ι
      j : κ
      i' : ι
      j' : κ
      ⊢ Eq (HMul.hMul (HMul.hMul ((Finsupp.single j 1) j) ((Finsupp.single i 1) i))  …
    -/
  · simpa using mul_comm _ _
    /-
      🎉 no goals
    -/
  /-
    case mk.mk.h₀
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    b : Basis ι R M
    c : Basis κ R N
    i : ι
    j : κ
    i' : ι
    j' : κ
    ⊢ ∀ (b : κ), Membership.mem Finset.univ b → Ne b j → Eq (HMul.hMul (HMul.hMul  …
  -/
  all_goals { intros; simp [*] at * }
  /-
    🎉 no goals
  -/

-- Porting note: introduced to help with timeout in dualDistribEquivOfBasis

theorem dualDistrib_dualDistribInvOfBasis_right_inverse (b : Basis ι R M) (c : Basis κ R N) :
    comp (dualDistribInvOfBasis b c) (dualDistrib R M N) = LinearMap.id := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    b : Basis ι R M
    c : Basis κ R N
    ⊢ Eq ((TensorProduct.dualDistribInvOfBasis b c).comp (TensorProduct.dualDistri …
  -/
  apply (b.dualBasis.tensorProduct c.dualBasis).ext
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    b : Basis ι R M
    c : Basis κ R N
    ⊢ ∀ (i : Prod ι κ), Eq (((TensorProduct.dualDistribInvOfBasis b c).comp (Tenso …
  -/
  rintro ⟨i, j⟩
  simp only [Basis.tensorProduct_apply, Basis.coe_dualBasis, coe_comp, Function.comp_apply,
    dualDistribInvOfBasis_apply, dualDistrib_apply, Basis.coord_apply, Basis.repr_self,
    ne_eq, id_coe, id_eq]
  /-
    case mk
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    b : Basis ι R M
    c : Basis κ R N
    i : ι
    j : κ
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun x_1 => HSMul.hSMul (HMul.hM …
  -/
  rw [Finset.sum_eq_single i, Finset.sum_eq_single j]
    /-
      case mk
      R : Type u_1
      M : Type u_3
      N : Type u_4
      ι : Type u_5
      κ : Type u_6
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : DecidableEq κ
      inst✝⁶ : Fintype ι
      inst✝⁵ : Fintype κ
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      b : Basis ι R M
      c : Basis κ R N
      i : ι
      j : κ
      ⊢ Eq (HSMul.hSMul (HMul.hMul ((Finsupp.single i 1) i) ((Finsupp.single j 1) j) …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case mk.h₀
    R : Type u_1
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    b : Basis ι R M
    c : Basis κ R N
    i : ι
    j : κ
    ⊢ ∀ (b_1 : κ), Membership.mem Finset.univ b_1 → Ne b_1 j → Eq (HSMul.hSMul (HM …
  -/
  all_goals { intros; simp [*] at * }
  /-
    🎉 no goals
  -/


/-- A linear equivalence between `Dual M ⊗ Dual N` and `Dual (M ⊗ N)` given bases for `M` and `N`.
It sends `f ⊗ g` to the composition of `TensorProduct.map f g` with the natural
isomorphism `R ⊗ R ≃ R`.
-/
@[simps!]
noncomputable def dualDistribEquivOfBasis (b : Basis ι R M) (c : Basis κ R N) :
    Dual R M ⊗[R] Dual R N ≃ₗ[R] Dual R (M ⊗[R] N) := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : DecidableEq κ
    inst✝⁶ : Fintype ι
    inst✝⁵ : Fintype κ
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    b : Basis ι R M
    c : Basis κ R N
    ⊢ LinearEquiv (RingHom.id R) (TensorProduct R (Module.Dual R M) (Module.Dual R …
  -/
  refine LinearEquiv.ofLinear (dualDistrib R M N) (dualDistribInvOfBasis b c) ?_ ?_
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      M : Type u_3
      N : Type u_4
      ι : Type u_5
      κ : Type u_6
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : DecidableEq κ
      inst✝⁶ : Fintype ι
      inst✝⁵ : Fintype κ
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      b : Basis ι R M
      c : Basis κ R N
      ⊢ Eq ((TensorProduct.dualDistrib R M N).comp (TensorProduct.dualDistribInvOfBa …
    -/
  · exact dualDistrib_dualDistribInvOfBasis_left_inverse _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      M : Type u_3
      N : Type u_4
      ι : Type u_5
      κ : Type u_6
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : DecidableEq κ
      inst✝⁶ : Fintype ι
      inst✝⁵ : Fintype κ
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      b : Basis ι R M
      c : Basis κ R N
      ⊢ Eq ((TensorProduct.dualDistribInvOfBasis b c).comp (TensorProduct.dualDistri …
    -/
  · exact dualDistrib_dualDistribInvOfBasis_right_inverse _ _
    /-
      🎉 no goals
    -/


/--
A linear equivalence between `Dual M ⊗ Dual N` and `Dual (M ⊗ N)` when `M` and `N` are finite free
modules. It sends `f ⊗ g` to the composition of `TensorProduct.map f g` with the natural
isomorphism `R ⊗ R ≃ R`.
-/
@[simp]
noncomputable def dualDistribEquiv : Dual R M ⊗[R] Dual R N ≃ₗ[R] Dual R (M ⊗[R] N) :=
  dualDistribEquivOfBasis (Module.Free.chooseBasis R M) (Module.Free.chooseBasis R N)


