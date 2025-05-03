/-- The closed parallelepiped spanned by a finite family of vectors. -/
def parallelepiped (v : ι → E) : Set E :=
  (fun t : ι → ℝ => ∑ i, t i • v i) '' Icc 0 1


theorem mem_parallelepiped_iff (v : ι → E) (x : E) :
    x ∈ parallelepiped v ↔ ∃ t ∈ Icc (0 : ι → ℝ) 1, x = ∑ i, t i • v i := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    x : E
    ⊢ Iff (Membership.mem (parallelepiped v) x) (Exists fun t => And (Membership.m …
  -/
  simp [parallelepiped, eq_comm]
  /-
    🎉 no goals
  -/


theorem parallelepiped_basis_eq (b : Basis ι ℝ E) :
    parallelepiped b = {x | ∀ i, b.repr x i ∈ Set.Icc 0 1} := by
  classical
  ext x
  simp_rw [mem_parallelepiped_iff, mem_setOf_eq, b.ext_elem_iff, _root_.map_sum,
    _root_.map_smul, Finset.sum_apply', Basis.repr_self, Finsupp.smul_single, smul_eq_mul,
    mul_one, Finsupp.single_apply, Finset.sum_ite_eq', Finset.mem_univ, ite_true, mem_Icc,
    Pi.le_def, Pi.zero_apply, Pi.one_apply, ← forall_and]
  aesop


theorem image_parallelepiped (f : E →ₗ[ℝ] F) (v : ι → E) :
    f '' parallelepiped v = parallelepiped (f ∘ v) := by
  /-
    ι : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : Fintype ι
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : AddCommGroup F
    inst✝ : Module Real F
    f : LinearMap (RingHom.id Real) E F
    v : ι → E
    ⊢ Eq (Set.image (⇑f) (parallelepiped v)) (parallelepiped (Function.comp (⇑f) v))
  -/
  simp only [parallelepiped, ← image_comp]
  /-
    ι : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : Fintype ι
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : AddCommGroup F
    inst✝ : Module Real F
    f : LinearMap (RingHom.id Real) E F
    v : ι → E
    ⊢ Eq (Set.image (Function.comp ⇑f fun t => Finset.univ.sum fun i => HSMul.hSMu …
  -/
  congr 1 with t
  /-
    case h
    ι : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : Fintype ι
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : AddCommGroup F
    inst✝ : Module Real F
    f : LinearMap (RingHom.id Real) E F
    v : ι → E
    t : F
    ⊢ Iff (Membership.mem (Set.image (Function.comp ⇑f fun t => Finset.univ.sum fu …
  -/
  simp only [Function.comp_apply, _root_.map_sum, LinearMap.map_smulₛₗ, RingHom.id_apply]
  /-
    🎉 no goals
  -/


/-- Reindexing a family of vectors does not change their parallelepiped. -/
@[simp]
theorem parallelepiped_comp_equiv (v : ι → E) (e : ι' ≃ ι) :
    parallelepiped (v ∘ e) = parallelepiped v := by
  /-
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    inst✝³ : Fintype ι
    inst✝² : Fintype ι'
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    e : Equiv ι' ι
    ⊢ Eq (parallelepiped (Function.comp v ⇑e)) (parallelepiped v)
  -/
  simp only [parallelepiped]
  /-
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    inst✝³ : Fintype ι
    inst✝² : Fintype ι'
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    e : Equiv ι' ι
    ⊢ Eq (Set.image (fun t => Finset.univ.sum fun i => HSMul.hSMul (t i) (Function …
  -/
  let K : (ι' → ℝ) ≃ (ι → ℝ) := Equiv.piCongrLeft' (fun _a : ι' => ℝ) e
  have : Icc (0 : ι → ℝ) 1 = K '' Icc (0 : ι' → ℝ) 1 := by
    rw [← Equiv.preimage_eq_iff_eq_image]
    ext x
    simp only [K, mem_preimage, mem_Icc, Pi.le_def, Pi.zero_apply, Equiv.piCongrLeft'_apply,
      Pi.one_apply]
    refine
      ⟨fun h => ⟨fun i => ?_, fun i => ?_⟩, fun h =>
        ⟨fun i => h.1 (e.symm i), fun i => h.2 (e.symm i)⟩⟩
    · simpa only [Equiv.symm_apply_apply] using h.1 (e i)
    · simpa only [Equiv.symm_apply_apply] using h.2 (e i)
  /-
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    inst✝³ : Fintype ι
    inst✝² : Fintype ι'
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    e : Equiv ι' ι
    K : Equiv (ι' → Real) (ι → Real) := Equiv.piCongrLeft' (fun _a => Real) e
    this : Eq (Set.Icc 0 1) (Set.image (⇑K) (Set.Icc 0 1))
    ⊢ Eq (Set.image (fun t => Finset.univ.sum fun i => HSMul.hSMul (t i) (Function …
  -/
  rw [this, ← image_comp]
  /-
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    inst✝³ : Fintype ι
    inst✝² : Fintype ι'
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    e : Equiv ι' ι
    K : Equiv (ι' → Real) (ι → Real) := Equiv.piCongrLeft' (fun _a => Real) e
    this : Eq (Set.Icc 0 1) (Set.image (⇑K) (Set.Icc 0 1))
    ⊢ Eq (Set.image (fun t => Finset.univ.sum fun i => HSMul.hSMul (t i) (Function …
  -/
  congr 1 with x
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    inst✝³ : Fintype ι
    inst✝² : Fintype ι'
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    e : Equiv ι' ι
    K : Equiv (ι' → Real) (ι → Real) := Equiv.piCongrLeft' (fun _a => Real) e
    this : Eq (Set.Icc 0 1) (Set.image (⇑K) (Set.Icc 0 1))
    x : E
    ⊢ Iff (Membership.mem (Set.image (fun t => Finset.univ.sum fun i => HSMul.hSMu …
  -/
  have := fun z : ι' → ℝ => e.symm.sum_comp fun i => z i • v (e i)
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    inst✝³ : Fintype ι
    inst✝² : Fintype ι'
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    e : Equiv ι' ι
    K : Equiv (ι' → Real) (ι → Real) := Equiv.piCongrLeft' (fun _a => Real) e
    this✝ : Eq (Set.Icc 0 1) (Set.image (⇑K) (Set.Icc 0 1))
    x : E
    this : ∀ (z : ι' → Real), Eq (Finset.univ.sum fun i => HSMul.hSMul (z (e.symm  …
    ⊢ Iff (Membership.mem (Set.image (fun t => Finset.univ.sum fun i => HSMul.hSMu …
  -/
  simp_rw [Equiv.apply_symm_apply] at this
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    inst✝³ : Fintype ι
    inst✝² : Fintype ι'
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    e : Equiv ι' ι
    K : Equiv (ι' → Real) (ι → Real) := Equiv.piCongrLeft' (fun _a => Real) e
    this✝ : Eq (Set.Icc 0 1) (Set.image (⇑K) (Set.Icc 0 1))
    x : E
    this : ∀ (z : ι' → Real), Eq (Finset.univ.sum fun x => HSMul.hSMul (z (e.symm  …
    ⊢ Iff (Membership.mem (Set.image (fun t => Finset.univ.sum fun i => HSMul.hSMu …
  -/
  simp_rw [Function.comp_apply, mem_image, mem_Icc, K, Equiv.piCongrLeft'_apply, this]
  /-
    🎉 no goals
  -/

-- The parallelepiped associated to an orthonormal basis of `ℝ` is either `[0, 1]` or `[-1, 0]`.

theorem parallelepiped_orthonormalBasis_one_dim (b : OrthonormalBasis ι ℝ ℝ) :
    parallelepiped b = Icc 0 1 ∨ parallelepiped b = Icc (-1) 0 := by
  have e : ι ≃ Fin 1 := by
    apply Fintype.equivFinOfCardEq
    simp only [← finrank_eq_card_basis b.toBasis, finrank_self]
  have B : parallelepiped (b.reindex e) = parallelepiped b := by
    convert parallelepiped_comp_equiv b e.symm
    ext i
    simp only [OrthonormalBasis.coe_reindex]
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    b : OrthonormalBasis ι Real Real
    e : Equiv ι (Fin 1)
    B : Eq (parallelepiped ⇑(b.reindex e)) (parallelepiped ⇑b)
    ⊢ Or (Eq (parallelepiped ⇑b) (Set.Icc 0 1)) (Eq (parallelepiped ⇑b) (Set.Icc ( …
  -/
  rw [← B]
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    b : OrthonormalBasis ι Real Real
    e : Equiv ι (Fin 1)
    B : Eq (parallelepiped ⇑(b.reindex e)) (parallelepiped ⇑b)
    ⊢ Or (Eq (parallelepiped ⇑(b.reindex e)) (Set.Icc 0 1)) (Eq (parallelepiped ⇑( …
  -/
  let F : ℝ → Fin 1 → ℝ := fun t => fun _i => t
  have A : Icc (0 : Fin 1 → ℝ) 1 = F '' Icc (0 : ℝ) 1 := by
    apply Subset.antisymm
    · intro x hx
      refine ⟨x 0, ⟨hx.1 0, hx.2 0⟩, ?_⟩
      ext j
      simp only [F, Subsingleton.elim j 0]
    · rintro x ⟨y, hy, rfl⟩
      exact ⟨fun _j => hy.1, fun _j => hy.2⟩
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    b : OrthonormalBasis ι Real Real
    e : Equiv ι (Fin 1)
    B : Eq (parallelepiped ⇑(b.reindex e)) (parallelepiped ⇑b)
    F : Real → Fin 1 → Real := fun t _i => t
    A : Eq (Set.Icc 0 1) (Set.image F (Set.Icc 0 1))
    ⊢ Or (Eq (parallelepiped ⇑(b.reindex e)) (Set.Icc 0 1)) (Eq (parallelepiped ⇑( …
  -/
  rcases orthonormalBasis_one_dim (b.reindex e) with (H | H)
    /-
      case inl
      ι : Type u_1
      inst✝ : Fintype ι
      b : OrthonormalBasis ι Real Real
      e : Equiv ι (Fin 1)
      B : Eq (parallelepiped ⇑(b.reindex e)) (parallelepiped ⇑b)
      F : Real → Fin 1 → Real := fun t _i => t
      A : Eq (Set.Icc 0 1) (Set.image F (Set.Icc 0 1))
      H : Eq ⇑(b.reindex e) fun x => 1
      ⊢ Or (Eq (parallelepiped ⇑(b.reindex e)) (Set.Icc 0 1)) (Eq (parallelepiped ⇑( …
    -/
  · left
    /-
      case inl.h
      ι : Type u_1
      inst✝ : Fintype ι
      b : OrthonormalBasis ι Real Real
      e : Equiv ι (Fin 1)
      B : Eq (parallelepiped ⇑(b.reindex e)) (parallelepiped ⇑b)
      F : Real → Fin 1 → Real := fun t _i => t
      A : Eq (Set.Icc 0 1) (Set.image F (Set.Icc 0 1))
      H : Eq ⇑(b.reindex e) fun x => 1
      ⊢ Eq (parallelepiped ⇑(b.reindex e)) (Set.Icc 0 1)
    -/
    simp_rw [parallelepiped, H, A, Algebra.id.smul_eq_mul, mul_one]
    simp only [F, Finset.univ_unique, Fin.default_eq_zero, Finset.sum_singleton,
      ← image_comp, Function.comp_apply, image_id']
    /-
      case inr
      ι : Type u_1
      inst✝ : Fintype ι
      b : OrthonormalBasis ι Real Real
      e : Equiv ι (Fin 1)
      B : Eq (parallelepiped ⇑(b.reindex e)) (parallelepiped ⇑b)
      F : Real → Fin 1 → Real := fun t _i => t
      A : Eq (Set.Icc 0 1) (Set.image F (Set.Icc 0 1))
      H : Eq ⇑(b.reindex e) fun x => -1
      ⊢ Or (Eq (parallelepiped ⇑(b.reindex e)) (Set.Icc 0 1)) (Eq (parallelepiped ⇑( …
    -/
  · right
    /-
      case inr.h
      ι : Type u_1
      inst✝ : Fintype ι
      b : OrthonormalBasis ι Real Real
      e : Equiv ι (Fin 1)
      B : Eq (parallelepiped ⇑(b.reindex e)) (parallelepiped ⇑b)
      F : Real → Fin 1 → Real := fun t _i => t
      A : Eq (Set.Icc 0 1) (Set.image F (Set.Icc 0 1))
      H : Eq ⇑(b.reindex e) fun x => -1
      ⊢ Eq (parallelepiped ⇑(b.reindex e)) (Set.Icc (-1) 0)
    -/
    simp_rw [H, parallelepiped, Algebra.id.smul_eq_mul, A]
    simp only [F, Finset.univ_unique, Fin.default_eq_zero, mul_neg, mul_one, Finset.sum_neg_distrib,
      Finset.sum_singleton, ← image_comp, Function.comp, image_neg_eq_neg, neg_Icc, neg_zero]


theorem parallelepiped_eq_sum_segment (v : ι → E) : parallelepiped v = ∑ i, segment ℝ 0 (v i) := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    ⊢ Eq (parallelepiped v) (Finset.univ.sum fun i => segment Real 0 (v i))
  -/
  ext
  simp only [mem_parallelepiped_iff, Set.mem_finset_sum, Finset.mem_univ, forall_true_left,
    segment_eq_image, smul_zero, zero_add, ← Set.pi_univ_Icc, Set.mem_univ_pi]
  /-
    case h
    ι : Type u_1
    E : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    x✝ : E
    ⊢ Iff (Exists fun t => And (∀ (i : ι), Membership.mem (Set.Icc (0 i) (1 i)) (t …
  -/
  constructor
    /-
      case h.mp
      ι : Type u_1
      E : Type u_3
      inst✝² : Fintype ι
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      v : ι → E
      x✝ : E
      ⊢ (Exists fun t => And (∀ (i : ι), Membership.mem (Set.Icc (0 i) (1 i)) (t i)) …
    -/
  · rintro ⟨t, ht, rfl⟩
    /-
      case h.mp.intro.intro
      ι : Type u_1
      E : Type u_3
      inst✝² : Fintype ι
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      v : ι → E
      t : ι → Real
      ht : ∀ (i : ι), Membership.mem (Set.Icc (0 i) (1 i)) (t i)
      ⊢ Exists fun g => Exists fun h => Eq (Finset.univ.sum fun i => g i) (Finset.un …
    -/
    exact ⟨t • v, fun {i} => ⟨t i, ht _, by simp⟩, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case h.mpr
    ι : Type u_1
    E : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    x✝ : E
    ⊢ (Exists fun g => Exists fun h => Eq (Finset.univ.sum fun i => g i) x✝) → Exi …
  -/
  rintro ⟨g, hg, rfl⟩
  /-
    case h.mpr.intro.intro
    ι : Type u_1
    E : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v g : ι → E
    hg : ∀ {i : ι}, Membership.mem (Set.image (fun a => HSMul.hSMul a (v i)) (Set. …
    ⊢ Exists fun t => And (∀ (i : ι), Membership.mem (Set.Icc (0 i) (1 i)) (t i))  …
  -/
  choose t ht hg using @hg
  /-
    case h.mpr.intro.intro
    ι : Type u_1
    E : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v g : ι → E
    t : {i : ι} → Real
    ht : ∀ {i : ι}, Membership.mem (Set.Icc 0 1) t
    hg : ∀ {i : ι}, Eq ((fun a => HSMul.hSMul a (v i)) t) (g i)
    ⊢ Exists fun t => And (∀ (i : ι), Membership.mem (Set.Icc (0 i) (1 i)) (t i))  …
  -/
  refine ⟨@t, @ht, ?_⟩
  /-
    case h.mpr.intro.intro
    ι : Type u_1
    E : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v g : ι → E
    t : {i : ι} → Real
    ht : ∀ {i : ι}, Membership.mem (Set.Icc 0 1) t
    hg : ∀ {i : ι}, Eq ((fun a => HSMul.hSMul a (v i)) t) (g i)
    ⊢ Eq (Finset.univ.sum fun i => g i) (Finset.univ.sum fun i => HSMul.hSMul t (v …
  -/
  simp_rw [hg]
  /-
    🎉 no goals
  -/


theorem convex_parallelepiped (v : ι → E) : Convex ℝ (parallelepiped v) := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    ⊢ Convex Real (parallelepiped v)
  -/
  rw [parallelepiped_eq_sum_segment]
  /-
    ι : Type u_1
    E : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    ⊢ Convex Real (Finset.univ.sum fun i => segment Real 0 (v i))
  -/
  exact convex_sum _ fun _i _hi => convex_segment _ _
  /-
    🎉 no goals
  -/


/-- A `parallelepiped` is the convex hull of its vertices -/
theorem parallelepiped_eq_convexHull (v : ι → E) :
    parallelepiped v = convexHull ℝ (∑ i, {(0 : E), v i}) := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    v : ι → E
    ⊢ Eq (parallelepiped v) ((convexHull Real) (Finset.univ.sum fun i => Insert.in …
  -/
  simp_rw [convexHull_sum, convexHull_pair, parallelepiped_eq_sum_segment]
  /-
    🎉 no goals
  -/


/-- The axis aligned parallelepiped over `ι → ℝ` is a cuboid. -/
theorem parallelepiped_single [DecidableEq ι] (a : ι → ℝ) :
    (parallelepiped fun i => Pi.single i (a i)) = Set.uIcc 0 a := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    a : ι → Real
    ⊢ Eq (parallelepiped fun i => Pi.single i (a i)) (Set.uIcc 0 a)
  -/
  ext x
  simp_rw [Set.uIcc, mem_parallelepiped_iff, Set.mem_Icc, Pi.le_def, ← forall_and, Pi.inf_apply,
    Pi.sup_apply, ← Pi.single_smul', Pi.one_apply, Pi.zero_apply, ← Pi.smul_apply',
    Finset.univ_sum_single (_ : ι → ℝ)]
  /-
    case h
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    a x : ι → Real
    ⊢ Iff (Exists fun t => And (∀ (x : ι), And (LE.le 0 (t x)) (LE.le (t x) 1)) (E …
  -/
  constructor
    /-
      case h.mp
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      a x : ι → Real
      ⊢ (Exists fun t => And (∀ (x : ι), And (LE.le 0 (t x)) (LE.le (t x) 1)) (Eq x  …
    -/
  · rintro ⟨t, ht, rfl⟩ i
    /-
      case h.mp.intro.intro
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      a t : ι → Real
      ht : ∀ (x : ι), And (LE.le 0 (t x)) (LE.le (t x) 1)
      i : ι
      ⊢ And (LE.le (Min.min 0 (a i)) (HSMul.hSMul t a i)) (LE.le (HSMul.hSMul t a i) …
    -/
    specialize ht i
    /-
      case h.mp.intro.intro
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      a t : ι → Real
      i : ι
      ht : And (LE.le 0 (t i)) (LE.le (t i) 1)
      ⊢ And (LE.le (Min.min 0 (a i)) (HSMul.hSMul t a i)) (LE.le (HSMul.hSMul t a i) …
    -/
    simp_rw [smul_eq_mul, Pi.mul_apply]
    /-
      case h.mp.intro.intro
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      a t : ι → Real
      i : ι
      ht : And (LE.le 0 (t i)) (LE.le (t i) 1)
      ⊢ And (LE.le (Min.min 0 (a i)) (HMul.hMul (t i) (a i))) (LE.le (HMul.hMul (t i …
    -/
    rcases le_total (a i) 0 with hai | hai
      /-
        case h.mp.intro.intro.inl
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        a t : ι → Real
        i : ι
        ht : And (LE.le 0 (t i)) (LE.le (t i) 1)
        hai : LE.le (a i) 0
        ⊢ And (LE.le (Min.min 0 (a i)) (HMul.hMul (t i) (a i))) (LE.le (HMul.hMul (t i …
      -/
    · rw [sup_eq_left.mpr hai, inf_eq_right.mpr hai]
      /-
        case h.mp.intro.intro.inl
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        a t : ι → Real
        i : ι
        ht : And (LE.le 0 (t i)) (LE.le (t i) 1)
        hai : LE.le (a i) 0
        ⊢ And (LE.le (a i) (HMul.hMul (t i) (a i))) (LE.le (HMul.hMul (t i) (a i)) 0)
      -/
      exact ⟨le_mul_of_le_one_left hai ht.2, mul_nonpos_of_nonneg_of_nonpos ht.1 hai⟩
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.inr
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        a t : ι → Real
        i : ι
        ht : And (LE.le 0 (t i)) (LE.le (t i) 1)
        hai : LE.le 0 (a i)
        ⊢ And (LE.le (Min.min 0 (a i)) (HMul.hMul (t i) (a i))) (LE.le (HMul.hMul (t i …
      -/
    · rw [sup_eq_right.mpr hai, inf_eq_left.mpr hai]
      /-
        case h.mp.intro.intro.inr
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        a t : ι → Real
        i : ι
        ht : And (LE.le 0 (t i)) (LE.le (t i) 1)
        hai : LE.le 0 (a i)
        ⊢ And (LE.le 0 (HMul.hMul (t i) (a i))) (LE.le (HMul.hMul (t i) (a i)) (a i))
      -/
      exact ⟨mul_nonneg ht.1 hai, mul_le_of_le_one_left hai ht.2⟩
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      a x : ι → Real
      ⊢ (∀ (x_1 : ι), And (LE.le (Min.min 0 (a x_1)) (x x_1)) (LE.le (x x_1) (Max.ma …
    -/
  · intro h
    /-
      case h.mpr
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      a x : ι → Real
      h : ∀ (x_1 : ι), And (LE.le (Min.min 0 (a x_1)) (x x_1)) (LE.le (x x_1) (Max.m …
      ⊢ Exists fun t => And (∀ (x : ι), And (LE.le 0 (t x)) (LE.le (t x) 1)) (Eq x ( …
    -/
    refine ⟨fun i => x i / a i, fun i => ?_, funext fun i => ?_⟩
      /-
        case h.mpr.refine_1
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        a x : ι → Real
        h : ∀ (x_1 : ι), And (LE.le (Min.min 0 (a x_1)) (x x_1)) (LE.le (x x_1) (Max.m …
        i : ι
        ⊢ And (LE.le 0 ((fun i => HDiv.hDiv (x i) (a i)) i)) (LE.le ((fun i => HDiv.hD …
      -/
    · specialize h i
      /-
        case h.mpr.refine_1
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        a x : ι → Real
        i : ι
        h : And (LE.le (Min.min 0 (a i)) (x i)) (LE.le (x i) (Max.max 0 (a i)))
        ⊢ And (LE.le 0 ((fun i => HDiv.hDiv (x i) (a i)) i)) (LE.le ((fun i => HDiv.hD …
      -/
      rcases le_total (a i) 0 with hai | hai
        /-
          case h.mpr.refine_1.inl
          ι : Type u_1
          inst✝¹ : Fintype ι
          inst✝ : DecidableEq ι
          a x : ι → Real
          i : ι
          h : And (LE.le (Min.min 0 (a i)) (x i)) (LE.le (x i) (Max.max 0 (a i)))
          hai : LE.le (a i) 0
          ⊢ And (LE.le 0 ((fun i => HDiv.hDiv (x i) (a i)) i)) (LE.le ((fun i => HDiv.hD …
        -/
      · rw [sup_eq_left.mpr hai, inf_eq_right.mpr hai] at h
        /-
          case h.mpr.refine_1.inl
          ι : Type u_1
          inst✝¹ : Fintype ι
          inst✝ : DecidableEq ι
          a x : ι → Real
          i : ι
          h : And (LE.le (a i) (x i)) (LE.le (x i) 0)
          hai : LE.le (a i) 0
          ⊢ And (LE.le 0 ((fun i => HDiv.hDiv (x i) (a i)) i)) (LE.le ((fun i => HDiv.hD …
        -/
        exact ⟨div_nonneg_of_nonpos h.2 hai, div_le_one_of_ge h.1 hai⟩
        /-
          🎉 no goals
        -/
        /-
          case h.mpr.refine_1.inr
          ι : Type u_1
          inst✝¹ : Fintype ι
          inst✝ : DecidableEq ι
          a x : ι → Real
          i : ι
          h : And (LE.le (Min.min 0 (a i)) (x i)) (LE.le (x i) (Max.max 0 (a i)))
          hai : LE.le 0 (a i)
          ⊢ And (LE.le 0 ((fun i => HDiv.hDiv (x i) (a i)) i)) (LE.le ((fun i => HDiv.hD …
        -/
      · rw [sup_eq_right.mpr hai, inf_eq_left.mpr hai] at h
        /-
          case h.mpr.refine_1.inr
          ι : Type u_1
          inst✝¹ : Fintype ι
          inst✝ : DecidableEq ι
          a x : ι → Real
          i : ι
          h : And (LE.le 0 (x i)) (LE.le (x i) (a i))
          hai : LE.le 0 (a i)
          ⊢ And (LE.le 0 ((fun i => HDiv.hDiv (x i) (a i)) i)) (LE.le ((fun i => HDiv.hD …
        -/
        exact ⟨div_nonneg h.1 hai, div_le_one_of_le₀ h.2 hai⟩
        /-
          🎉 no goals
        -/
      /-
        case h.mpr.refine_2
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        a x : ι → Real
        h : ∀ (x_1 : ι), And (LE.le (Min.min 0 (a x_1)) (x x_1)) (LE.le (x x_1) (Max.m …
        i : ι
        ⊢ Eq (x i) (HSMul.hSMul (fun i => HDiv.hDiv (x i) (a i)) a i)
      -/
    · specialize h i
      /-
        case h.mpr.refine_2
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        a x : ι → Real
        i : ι
        h : And (LE.le (Min.min 0 (a i)) (x i)) (LE.le (x i) (Max.max 0 (a i)))
        ⊢ Eq (x i) (HSMul.hSMul (fun i => HDiv.hDiv (x i) (a i)) a i)
      -/
      simp only [smul_eq_mul, Pi.mul_apply]
      /-
        case h.mpr.refine_2
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        a x : ι → Real
        i : ι
        h : And (LE.le (Min.min 0 (a i)) (x i)) (LE.le (x i) (Max.max 0 (a i)))
        ⊢ Eq (x i) (HMul.hMul (HDiv.hDiv (x i) (a i)) (a i))
      -/
      rcases eq_or_ne (a i) 0 with hai | hai
        /-
          case h.mpr.refine_2.inl
          ι : Type u_1
          inst✝¹ : Fintype ι
          inst✝ : DecidableEq ι
          a x : ι → Real
          i : ι
          h : And (LE.le (Min.min 0 (a i)) (x i)) (LE.le (x i) (Max.max 0 (a i)))
          hai : Eq (a i) 0
          ⊢ Eq (x i) (HMul.hMul (HDiv.hDiv (x i) (a i)) (a i))
        -/
      · rw [hai, inf_idem, sup_idem, ← le_antisymm_iff] at h
        /-
          case h.mpr.refine_2.inl
          ι : Type u_1
          inst✝¹ : Fintype ι
          inst✝ : DecidableEq ι
          a x : ι → Real
          i : ι
          h : Eq 0 (x i)
          hai : Eq (a i) 0
          ⊢ Eq (x i) (HMul.hMul (HDiv.hDiv (x i) (a i)) (a i))
        -/
        rw [hai, ← h, zero_div, zero_mul]
        /-
          🎉 no goals
        -/
        /-
          case h.mpr.refine_2.inr
          ι : Type u_1
          inst✝¹ : Fintype ι
          inst✝ : DecidableEq ι
          a x : ι → Real
          i : ι
          h : And (LE.le (Min.min 0 (a i)) (x i)) (LE.le (x i) (Max.max 0 (a i)))
          hai : Ne (a i) 0
          ⊢ Eq (x i) (HMul.hMul (HDiv.hDiv (x i) (a i)) (a i))
        -/
      · rw [div_mul_cancel₀ _ hai]
        /-
          🎉 no goals
        -/


/-- The parallelepiped spanned by a basis, as a compact set with nonempty interior. -/
def Basis.parallelepiped (b : Basis ι ℝ E) : PositiveCompacts E where
  carrier := _root_.parallelepiped b
  isCompact' := IsCompact.image isCompact_Icc
      (continuous_finset_sum Finset.univ fun (i : ι) (_H : i ∈ Finset.univ) =>
        (continuous_apply i).smul continuous_const)
  interior_nonempty' := by
    suffices H : Set.Nonempty (interior (b.equivFunL.symm.toHomeomorph '' Icc 0 1)) by
      dsimp only [_root_.parallelepiped]
      convert H
      exact (b.equivFun_symm_apply _).symm
    have A : Set.Nonempty (interior (Icc (0 : ι → ℝ) 1)) := by
      rw [← pi_univ_Icc, interior_pi_set (@finite_univ ι _)]
      simp only [univ_pi_nonempty_iff, Pi.zero_apply, Pi.one_apply, interior_Icc, nonempty_Ioo,
        zero_lt_one, imp_true_iff]
    /-
      ι : Type u_1
      ι' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁵ : Fintype ι
      inst✝⁴ : Fintype ι'
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Real F
      b : Basis ι Real E
      A : (interior (Set.Icc 0 1)).Nonempty
      ⊢ (interior (Set.image (⇑b.equivFunL.symm.toHomeomorph) (Set.Icc 0 1))).Nonempty
    -/
    rwa [← Homeomorph.image_interior, image_nonempty]
    /-
      🎉 no goals
    -/


@[simp]
theorem Basis.coe_parallelepiped (b : Basis ι ℝ E) :
    (b.parallelepiped : Set E) = _root_.parallelepiped b := rfl


@[simp]
theorem Basis.parallelepiped_reindex (b : Basis ι ℝ E) (e : ι ≃ ι') :
    (b.reindex e).parallelepiped = b.parallelepiped :=
  PositiveCompacts.ext <|
    (congr_arg _root_.parallelepiped (b.coe_reindex e)).trans (parallelepiped_comp_equiv b e.symm)


theorem Basis.parallelepiped_map (b : Basis ι ℝ E) (e : E ≃ₗ[ℝ] F) :
    (b.map e).parallelepiped = b.parallelepiped.map e
    (have := FiniteDimensional.of_fintype_basis b
    -- Porting note: Lean cannot infer the instance above
    LinearMap.continuous_of_finiteDimensional e.toLinearMap)
    (have := FiniteDimensional.of_fintype_basis (b.map e)
    -- Porting note: Lean cannot infer the instance above
    LinearMap.isOpenMap_of_finiteDimensional _ e.surjective) :=
  PositiveCompacts.ext (image_parallelepiped e.toLinearMap _).symm


theorem Basis.prod_parallelepiped (v : Basis ι ℝ E) (w : Basis ι' ℝ F) :
    (v.prod w).parallelepiped = v.parallelepiped.prod w.parallelepiped := by
  /-
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Real F
    v : Basis ι Real E
    w : Basis ι' Real F
    ⊢ Eq (v.prod w).parallelepiped (v.parallelepiped.prod w.parallelepiped)
  -/
  ext x
  simp only [Basis.coe_parallelepiped, TopologicalSpace.PositiveCompacts.coe_prod, Set.mem_prod,
    mem_parallelepiped_iff]
  /-
    case h.h
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Real F
    v : Basis ι Real E
    w : Basis ι' Real F
    x : Prod E F
    ⊢ Iff (Exists fun t => And (Membership.mem (Set.Icc 0 1) t) (Eq x (Finset.univ …
  -/
  constructor
    /-
      case h.h.mp
      ι : Type u_1
      ι' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁵ : Fintype ι
      inst✝⁴ : Fintype ι'
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Real F
      v : Basis ι Real E
      w : Basis ι' Real F
      x : Prod E F
      ⊢ (Exists fun t => And (Membership.mem (Set.Icc 0 1) t) (Eq x (Finset.univ.sum …
    -/
  · intro h
    /-
      case h.h.mp
      ι : Type u_1
      ι' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁵ : Fintype ι
      inst✝⁴ : Fintype ι'
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Real F
      v : Basis ι Real E
      w : Basis ι' Real F
      x : Prod E F
      h : Exists fun t => And (Membership.mem (Set.Icc 0 1) t) (Eq x (Finset.univ.su …
      ⊢ And (Exists fun t => And (Membership.mem (Set.Icc 0 1) t) (Eq x.1 (Finset.un …
    -/
    rcases h with ⟨t, ht1, ht2⟩
    /-
      case h.h.mp.intro.intro
      ι : Type u_1
      ι' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁵ : Fintype ι
      inst✝⁴ : Fintype ι'
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Real F
      v : Basis ι Real E
      w : Basis ι' Real F
      x : Prod E F
      t : Sum ι ι' → Real
      ht1 : Membership.mem (Set.Icc 0 1) t
      ht2 : Eq x (Finset.univ.sum fun i => HSMul.hSMul (t i) ((v.prod w) i))
      ⊢ And (Exists fun t => And (Membership.mem (Set.Icc 0 1) t) (Eq x.1 (Finset.un …
    -/
    constructor
      /-
        case h.h.mp.intro.intro.left
        ι : Type u_1
        ι' : Type u_2
        E : Type u_3
        F : Type u_4
        inst✝⁵ : Fintype ι
        inst✝⁴ : Fintype ι'
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real E
        inst✝ : NormedSpace Real F
        v : Basis ι Real E
        w : Basis ι' Real F
        x : Prod E F
        t : Sum ι ι' → Real
        ht1 : Membership.mem (Set.Icc 0 1) t
        ht2 : Eq x (Finset.univ.sum fun i => HSMul.hSMul (t i) ((v.prod w) i))
        ⊢ Exists fun t => And (Membership.mem (Set.Icc 0 1) t) (Eq x.1 (Finset.univ.su …
      -/
    · use t ∘ Sum.inl
      /-
        case h
        ι : Type u_1
        ι' : Type u_2
        E : Type u_3
        F : Type u_4
        inst✝⁵ : Fintype ι
        inst✝⁴ : Fintype ι'
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real E
        inst✝ : NormedSpace Real F
        v : Basis ι Real E
        w : Basis ι' Real F
        x : Prod E F
        t : Sum ι ι' → Real
        ht1 : Membership.mem (Set.Icc 0 1) t
        ht2 : Eq x (Finset.univ.sum fun i => HSMul.hSMul (t i) ((v.prod w) i))
        ⊢ And (Membership.mem (Set.Icc 0 1) (Function.comp t Sum.inl)) (Eq x.1 (Finset …
      -/
      constructor
        /-
          case h.left
          ι : Type u_1
          ι' : Type u_2
          E : Type u_3
          F : Type u_4
          inst✝⁵ : Fintype ι
          inst✝⁴ : Fintype ι'
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedAddCommGroup F
          inst✝¹ : NormedSpace Real E
          inst✝ : NormedSpace Real F
          v : Basis ι Real E
          w : Basis ι' Real F
          x : Prod E F
          t : Sum ι ι' → Real
          ht1 : Membership.mem (Set.Icc 0 1) t
          ht2 : Eq x (Finset.univ.sum fun i => HSMul.hSMul (t i) ((v.prod w) i))
          ⊢ Membership.mem (Set.Icc 0 1) (Function.comp t Sum.inl)
        -/
      · exact ⟨(ht1.1 <| Sum.inl ·), (ht1.2 <| Sum.inl ·)⟩
        /-
          🎉 no goals
        -/
      /-
        case h.right
        ι : Type u_1
        ι' : Type u_2
        E : Type u_3
        F : Type u_4
        inst✝⁵ : Fintype ι
        inst✝⁴ : Fintype ι'
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real E
        inst✝ : NormedSpace Real F
        v : Basis ι Real E
        w : Basis ι' Real F
        x : Prod E F
        t : Sum ι ι' → Real
        ht1 : Membership.mem (Set.Icc 0 1) t
        ht2 : Eq x (Finset.univ.sum fun i => HSMul.hSMul (t i) ((v.prod w) i))
        ⊢ Eq x.1 (Finset.univ.sum fun i => HSMul.hSMul (Function.comp t Sum.inl i) (v  …
      -/
      simp [ht2, Prod.fst_sum, Prod.snd_sum]
      /-
        🎉 no goals
      -/
      /-
        case h.h.mp.intro.intro.right
        ι : Type u_1
        ι' : Type u_2
        E : Type u_3
        F : Type u_4
        inst✝⁵ : Fintype ι
        inst✝⁴ : Fintype ι'
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real E
        inst✝ : NormedSpace Real F
        v : Basis ι Real E
        w : Basis ι' Real F
        x : Prod E F
        t : Sum ι ι' → Real
        ht1 : Membership.mem (Set.Icc 0 1) t
        ht2 : Eq x (Finset.univ.sum fun i => HSMul.hSMul (t i) ((v.prod w) i))
        ⊢ Exists fun t => And (Membership.mem (Set.Icc 0 1) t) (Eq x.2 (Finset.univ.su …
      -/
    · use t ∘ Sum.inr
      /-
        case h
        ι : Type u_1
        ι' : Type u_2
        E : Type u_3
        F : Type u_4
        inst✝⁵ : Fintype ι
        inst✝⁴ : Fintype ι'
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real E
        inst✝ : NormedSpace Real F
        v : Basis ι Real E
        w : Basis ι' Real F
        x : Prod E F
        t : Sum ι ι' → Real
        ht1 : Membership.mem (Set.Icc 0 1) t
        ht2 : Eq x (Finset.univ.sum fun i => HSMul.hSMul (t i) ((v.prod w) i))
        ⊢ And (Membership.mem (Set.Icc 0 1) (Function.comp t Sum.inr)) (Eq x.2 (Finset …
      -/
      constructor
        /-
          case h.left
          ι : Type u_1
          ι' : Type u_2
          E : Type u_3
          F : Type u_4
          inst✝⁵ : Fintype ι
          inst✝⁴ : Fintype ι'
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedAddCommGroup F
          inst✝¹ : NormedSpace Real E
          inst✝ : NormedSpace Real F
          v : Basis ι Real E
          w : Basis ι' Real F
          x : Prod E F
          t : Sum ι ι' → Real
          ht1 : Membership.mem (Set.Icc 0 1) t
          ht2 : Eq x (Finset.univ.sum fun i => HSMul.hSMul (t i) ((v.prod w) i))
          ⊢ Membership.mem (Set.Icc 0 1) (Function.comp t Sum.inr)
        -/
      · exact ⟨(ht1.1 <| Sum.inr ·), (ht1.2 <| Sum.inr ·)⟩
        /-
          🎉 no goals
        -/
      /-
        case h.right
        ι : Type u_1
        ι' : Type u_2
        E : Type u_3
        F : Type u_4
        inst✝⁵ : Fintype ι
        inst✝⁴ : Fintype ι'
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real E
        inst✝ : NormedSpace Real F
        v : Basis ι Real E
        w : Basis ι' Real F
        x : Prod E F
        t : Sum ι ι' → Real
        ht1 : Membership.mem (Set.Icc 0 1) t
        ht2 : Eq x (Finset.univ.sum fun i => HSMul.hSMul (t i) ((v.prod w) i))
        ⊢ Eq x.2 (Finset.univ.sum fun i => HSMul.hSMul (Function.comp t Sum.inr i) (w  …
      -/
      simp [ht2, Prod.fst_sum, Prod.snd_sum]
      /-
        🎉 no goals
      -/
  /-
    case h.h.mpr
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Real F
    v : Basis ι Real E
    w : Basis ι' Real F
    x : Prod E F
    ⊢ And (Exists fun t => And (Membership.mem (Set.Icc 0 1) t) (Eq x.1 (Finset.un …
  -/
  intro h
  /-
    case h.h.mpr
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Real F
    v : Basis ι Real E
    w : Basis ι' Real F
    x : Prod E F
    h : And (Exists fun t => And (Membership.mem (Set.Icc 0 1) t) (Eq x.1 (Finset. …
    ⊢ Exists fun t => And (Membership.mem (Set.Icc 0 1) t) (Eq x (Finset.univ.sum  …
  -/
  rcases h with ⟨⟨t, ht1, ht2⟩, ⟨s, hs1, hs2⟩⟩
  /-
    case h.h.mpr.intro.intro.intro.intro.intro
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Real F
    v : Basis ι Real E
    w : Basis ι' Real F
    x : Prod E F
    t : ι → Real
    ht1 : Membership.mem (Set.Icc 0 1) t
    ht2 : Eq x.1 (Finset.univ.sum fun i => HSMul.hSMul (t i) (v i))
    s : ι' → Real
    hs1 : Membership.mem (Set.Icc 0 1) s
    hs2 : Eq x.2 (Finset.univ.sum fun i => HSMul.hSMul (s i) (w i))
    ⊢ Exists fun t => And (Membership.mem (Set.Icc 0 1) t) (Eq x (Finset.univ.sum  …
  -/
  use Sum.elim t s
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Real F
    v : Basis ι Real E
    w : Basis ι' Real F
    x : Prod E F
    t : ι → Real
    ht1 : Membership.mem (Set.Icc 0 1) t
    ht2 : Eq x.1 (Finset.univ.sum fun i => HSMul.hSMul (t i) (v i))
    s : ι' → Real
    hs1 : Membership.mem (Set.Icc 0 1) s
    hs2 : Eq x.2 (Finset.univ.sum fun i => HSMul.hSMul (s i) (w i))
    ⊢ And (Membership.mem (Set.Icc 0 1) (Sum.elim t s)) (Eq x (Finset.univ.sum fun …
  -/
  constructor
    /-
      case h.left
      ι : Type u_1
      ι' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁵ : Fintype ι
      inst✝⁴ : Fintype ι'
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Real F
      v : Basis ι Real E
      w : Basis ι' Real F
      x : Prod E F
      t : ι → Real
      ht1 : Membership.mem (Set.Icc 0 1) t
      ht2 : Eq x.1 (Finset.univ.sum fun i => HSMul.hSMul (t i) (v i))
      s : ι' → Real
      hs1 : Membership.mem (Set.Icc 0 1) s
      hs2 : Eq x.2 (Finset.univ.sum fun i => HSMul.hSMul (s i) (w i))
      ⊢ Membership.mem (Set.Icc 0 1) (Sum.elim t s)
    -/
  · constructor
      /-
        case h.left.left
        ι : Type u_1
        ι' : Type u_2
        E : Type u_3
        F : Type u_4
        inst✝⁵ : Fintype ι
        inst✝⁴ : Fintype ι'
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real E
        inst✝ : NormedSpace Real F
        v : Basis ι Real E
        w : Basis ι' Real F
        x : Prod E F
        t : ι → Real
        ht1 : Membership.mem (Set.Icc 0 1) t
        ht2 : Eq x.1 (Finset.univ.sum fun i => HSMul.hSMul (t i) (v i))
        s : ι' → Real
        hs1 : Membership.mem (Set.Icc 0 1) s
        hs2 : Eq x.2 (Finset.univ.sum fun i => HSMul.hSMul (s i) (w i))
        ⊢ LE.le 0 (Sum.elim t s)
      -/
    · change ∀ x : ι ⊕ ι', 0 ≤ Sum.elim t s x
      /-
        case h.left.left
        ι : Type u_1
        ι' : Type u_2
        E : Type u_3
        F : Type u_4
        inst✝⁵ : Fintype ι
        inst✝⁴ : Fintype ι'
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real E
        inst✝ : NormedSpace Real F
        v : Basis ι Real E
        w : Basis ι' Real F
        x : Prod E F
        t : ι → Real
        ht1 : Membership.mem (Set.Icc 0 1) t
        ht2 : Eq x.1 (Finset.univ.sum fun i => HSMul.hSMul (t i) (v i))
        s : ι' → Real
        hs1 : Membership.mem (Set.Icc 0 1) s
        hs2 : Eq x.2 (Finset.univ.sum fun i => HSMul.hSMul (s i) (w i))
        ⊢ ∀ (x : Sum ι ι'), LE.le 0 (Sum.elim t s x)
      -/
      aesop
      /-
        🎉 no goals
      -/
      /-
        case h.left.right
        ι : Type u_1
        ι' : Type u_2
        E : Type u_3
        F : Type u_4
        inst✝⁵ : Fintype ι
        inst✝⁴ : Fintype ι'
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real E
        inst✝ : NormedSpace Real F
        v : Basis ι Real E
        w : Basis ι' Real F
        x : Prod E F
        t : ι → Real
        ht1 : Membership.mem (Set.Icc 0 1) t
        ht2 : Eq x.1 (Finset.univ.sum fun i => HSMul.hSMul (t i) (v i))
        s : ι' → Real
        hs1 : Membership.mem (Set.Icc 0 1) s
        hs2 : Eq x.2 (Finset.univ.sum fun i => HSMul.hSMul (s i) (w i))
        ⊢ LE.le (Sum.elim t s) 1
      -/
    · change ∀ x : ι ⊕ ι', Sum.elim t s x ≤ 1
      /-
        case h.left.right
        ι : Type u_1
        ι' : Type u_2
        E : Type u_3
        F : Type u_4
        inst✝⁵ : Fintype ι
        inst✝⁴ : Fintype ι'
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real E
        inst✝ : NormedSpace Real F
        v : Basis ι Real E
        w : Basis ι' Real F
        x : Prod E F
        t : ι → Real
        ht1 : Membership.mem (Set.Icc 0 1) t
        ht2 : Eq x.1 (Finset.univ.sum fun i => HSMul.hSMul (t i) (v i))
        s : ι' → Real
        hs1 : Membership.mem (Set.Icc 0 1) s
        hs2 : Eq x.2 (Finset.univ.sum fun i => HSMul.hSMul (s i) (w i))
        ⊢ ∀ (x : Sum ι ι'), LE.le (Sum.elim t s x) 1
      -/
      aesop
      /-
        🎉 no goals
      -/
  /-
    case h.right
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Real F
    v : Basis ι Real E
    w : Basis ι' Real F
    x : Prod E F
    t : ι → Real
    ht1 : Membership.mem (Set.Icc 0 1) t
    ht2 : Eq x.1 (Finset.univ.sum fun i => HSMul.hSMul (t i) (v i))
    s : ι' → Real
    hs1 : Membership.mem (Set.Icc 0 1) s
    hs2 : Eq x.2 (Finset.univ.sum fun i => HSMul.hSMul (s i) (w i))
    ⊢ Eq x (Finset.univ.sum fun i => HSMul.hSMul (Sum.elim t s i) ((v.prod w) i))
  -/
  ext
    /-
      case h.right.fst
      ι : Type u_1
      ι' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁵ : Fintype ι
      inst✝⁴ : Fintype ι'
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Real F
      v : Basis ι Real E
      w : Basis ι' Real F
      x : Prod E F
      t : ι → Real
      ht1 : Membership.mem (Set.Icc 0 1) t
      ht2 : Eq x.1 (Finset.univ.sum fun i => HSMul.hSMul (t i) (v i))
      s : ι' → Real
      hs1 : Membership.mem (Set.Icc 0 1) s
      hs2 : Eq x.2 (Finset.univ.sum fun i => HSMul.hSMul (s i) (w i))
      ⊢ Eq x.1 (Finset.univ.sum fun i => HSMul.hSMul (Sum.elim t s i) ((v.prod w) i) …
    -/
  · simp [ht2, Prod.fst_sum]
    /-
      🎉 no goals
    -/
    /-
      case h.right.snd
      ι : Type u_1
      ι' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁵ : Fintype ι
      inst✝⁴ : Fintype ι'
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Real F
      v : Basis ι Real E
      w : Basis ι' Real F
      x : Prod E F
      t : ι → Real
      ht1 : Membership.mem (Set.Icc 0 1) t
      ht2 : Eq x.1 (Finset.univ.sum fun i => HSMul.hSMul (t i) (v i))
      s : ι' → Real
      hs1 : Membership.mem (Set.Icc 0 1) s
      hs2 : Eq x.2 (Finset.univ.sum fun i => HSMul.hSMul (s i) (w i))
      ⊢ Eq x.2 (Finset.univ.sum fun i => HSMul.hSMul (Sum.elim t s i) ((v.prod w) i) …
    -/
  · simp [hs2, Prod.snd_sum]
    /-
      🎉 no goals
    -/


/-- The Lebesgue measure associated to a basis, giving measure `1` to the parallelepiped spanned
by the basis. -/
irreducible_def Basis.addHaar (b : Basis ι ℝ E) : Measure E :=
  Measure.addHaarMeasure b.parallelepiped


instance IsAddHaarMeasure_basis_addHaar (b : Basis ι ℝ E) : IsAddHaarMeasure b.addHaar := by
  /-
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : Fintype ι'
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace Real F
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    b : Basis ι Real E
    ⊢ b.addHaar.IsAddHaarMeasure
  -/
  rw [Basis.addHaar]; exact Measure.isAddHaarMeasure_addHaarMeasure _
                      /-
                        🎉 no goals
                      -/


instance (b : Basis ι ℝ E) : SigmaFinite b.addHaar := by
  /-
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : Fintype ι'
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace Real F
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    b : Basis ι Real E
    ⊢ MeasureTheory.SigmaFinite b.addHaar
  -/
  have : FiniteDimensional ℝ E := FiniteDimensional.of_fintype_basis b
  /-
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : Fintype ι'
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace Real F
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    b : Basis ι Real E
    this : FiniteDimensional Real E
    ⊢ MeasureTheory.SigmaFinite b.addHaar
  -/
  rw [Basis.addHaar_def]; exact sigmaFinite_addHaarMeasure
                          /-
                            🎉 no goals
                          -/


/-- Let `μ` be a σ-finite left invariant measure on `E`. Then `μ` is equal to the Haar measure
defined by `b` iff the parallelepiped defined by `b` has measure `1` for `μ`. -/
theorem Basis.addHaar_eq_iff [SecondCountableTopology E] (b : Basis ι ℝ E) (μ : Measure E)
    [SigmaFinite μ] [IsAddLeftInvariant μ] :
    b.addHaar = μ ↔ μ b.parallelepiped = 1 := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝⁷ : Fintype ι
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : SecondCountableTopology E
    b : Basis ι Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : μ.IsAddLeftInvariant
    ⊢ Iff (Eq b.addHaar μ) (Eq (μ ↑b.parallelepiped) 1)
  -/
  rw [Basis.addHaar_def]
  /-
    ι : Type u_1
    E : Type u_3
    inst✝⁷ : Fintype ι
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : SecondCountableTopology E
    b : Basis ι Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : μ.IsAddLeftInvariant
    ⊢ Iff (Eq (MeasureTheory.Measure.addHaarMeasure b.parallelepiped) μ) (Eq (μ ↑b …
  -/
  exact addHaarMeasure_eq_iff b.parallelepiped μ
  /-
    🎉 no goals
  -/


@[simp]
theorem Basis.addHaar_reindex (b : Basis ι ℝ E) (e : ι ≃ ι') :
    (b.reindex e).addHaar = b.addHaar := by
  /-
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    inst✝⁵ : Fintype ι
    inst✝⁴ : Fintype ι'
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    b : Basis ι Real E
    e : Equiv ι ι'
    ⊢ Eq (b.reindex e).addHaar b.addHaar
  -/
  rw [Basis.addHaar, b.parallelepiped_reindex e, ← Basis.addHaar]
  /-
    🎉 no goals
  -/


theorem Basis.addHaar_self (b : Basis ι ℝ E) : b.addHaar (_root_.parallelepiped b) = 1 := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝⁴ : Fintype ι
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    b : Basis ι Real E
    ⊢ Eq (b.addHaar (_root_.parallelepiped ⇑b)) 1
  -/
  rw [Basis.addHaar]; exact addHaarMeasure_self
                      /-
                        🎉 no goals
                      -/


theorem Basis.prod_addHaar (v : Basis ι ℝ E) (w : Basis ι' ℝ F) :
    (v.prod w).addHaar = v.addHaar.prod w.addHaar := by
  /-
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : Fintype ι
    inst✝⁹ : Fintype ι'
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    inst✝ : SecondCountableTopologyEither E F
    v : Basis ι Real E
    w : Basis ι' Real F
    ⊢ Eq (v.prod w).addHaar (v.addHaar.prod w.addHaar)
  -/
  have : FiniteDimensional ℝ E := FiniteDimensional.of_fintype_basis v
  /-
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : Fintype ι
    inst✝⁹ : Fintype ι'
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    inst✝ : SecondCountableTopologyEither E F
    v : Basis ι Real E
    w : Basis ι' Real F
    this : FiniteDimensional Real E
    ⊢ Eq (v.prod w).addHaar (v.addHaar.prod w.addHaar)
  -/
  have : FiniteDimensional ℝ F := FiniteDimensional.of_fintype_basis w
  /-
    ι : Type u_1
    ι' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹⁰ : Fintype ι
    inst✝⁹ : Fintype ι'
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : NormedSpace Real F
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    inst✝ : SecondCountableTopologyEither E F
    v : Basis ι Real E
    w : Basis ι' Real F
    this✝ : FiniteDimensional Real E
    this : FiniteDimensional Real F
    ⊢ Eq (v.prod w).addHaar (v.addHaar.prod w.addHaar)
  -/
  simp [(v.prod w).addHaar_eq_iff, Basis.prod_parallelepiped, Basis.addHaar_self]
  /-
    🎉 no goals
  -/


/-- A finite dimensional inner product space has a canonical measure, the Lebesgue measure giving
volume `1` to the parallelepiped spanned by any orthonormal basis. We define the measure using
some arbitrary choice of orthonormal basis. The fact that it works with any orthonormal basis
is proved in `orthonormalBasis.volume_parallelepiped`.

This instance creates:

- a potential non-defeq diamond with the natural instance for `MeasureSpace (ULift E)`,
  which does not exist in Mathlib at the moment;

- a diamond with the existing instance `MeasureTheory.Measure.instMeasureSpacePUnit`.

However, we've decided not to refactor until one of these diamonds starts creating issues, see
https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/Hausdorff.20measure.20normalisation
-/
instance (priority := 100) measureSpaceOfInnerProductSpace [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E] :
    MeasureSpace E where volume := (stdOrthonormalBasis ℝ E).toBasis.addHaar


instance [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
    [MeasurableSpace E] [BorelSpace E] : IsAddHaarMeasure (volume : Measure E) :=
  IsAddHaarMeasure_basis_addHaar _

/- This instance should not be necessary, but Lean has difficulties to find it in product
situations if we do not declare it explicitly. -/

                                                  /-
                                                    ι : Type u_1
                                                    ι' : Type u_2
                                                    E : Type u_3
                                                    F : Type u_4
                                                    ⊢ MeasureTheory.MeasureSpace Real
                                                  -/
instance Real.measureSpace : MeasureSpace ℝ := by infer_instance
                                                  /-
                                                    🎉 no goals
                                                  -/


instance : MeasurableSpace (EuclideanSpace ℝ ι) := MeasurableSpace.pi


instance [Finite ι] : BorelSpace (EuclideanSpace ℝ ι) := Pi.borelSpace


/-- `WithLp.equiv` as a `MeasurableEquiv`. -/
@[simps toEquiv]
protected def measurableEquiv : EuclideanSpace ℝ ι ≃ᵐ (ι → ℝ) where
  toEquiv := WithLp.equiv _ _
  measurable_toFun := measurable_id
  measurable_invFun := measurable_id


theorem coe_measurableEquiv : ⇑(EuclideanSpace.measurableEquiv ι) = WithLp.equiv 2 _ := rfl


theorem coe_measurableEquiv_symm :
    ⇑(EuclideanSpace.measurableEquiv ι).symm = (WithLp.equiv 2 _).symm := rfl


