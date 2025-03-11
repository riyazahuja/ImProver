local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


/-- Induced inner product on a submodule. -/
instance Submodule.innerProductSpace (W : Submodule 𝕜 E) : InnerProductSpace 𝕜 W :=
  { Submodule.normedSpace W with
    inner := fun x y => ⟪(x : E), (y : E)⟫
    conj_symm := fun _ _ => inner_conj_symm _ _
    norm_sq_eq_inner := fun x => norm_sq_eq_inner (x : E)
    add_left := fun _ _ _ => inner_add_left _ _ _
    smul_left := fun _ _ _ => inner_smul_left _ _ _ }


/-- The inner product on submodules is the same as on the ambient space. -/
@[simp]
theorem Submodule.coe_inner (W : Submodule 𝕜 E) (x y : W) : ⟪x, y⟫ = ⟪(x : E), ↑y⟫ :=
  rfl


theorem Orthonormal.codRestrict {ι : Type*} {v : ι → E} (hv : Orthonormal 𝕜 v) (s : Submodule 𝕜 E)
    (hvs : ∀ i, v i ∈ s) : @Orthonormal 𝕜 s _ _ _ ι (Set.codRestrict v s hvs) :=
  s.subtypeₗᵢ.orthonormal_comp_iff.mp hv


theorem orthonormal_span {ι : Type*} {v : ι → E} (hv : Orthonormal 𝕜 v) :
    @Orthonormal 𝕜 (Submodule.span 𝕜 (Set.range v)) _ _ _ ι fun i : ι =>
      ⟨v i, Submodule.subset_span (Set.mem_range_self i)⟩ :=
  hv.codRestrict (Submodule.span 𝕜 (Set.range v)) fun i =>
    Submodule.subset_span (Set.mem_range_self i)


/-- An indexed family of mutually-orthogonal subspaces of an inner product space `E`.

The simple way to express this concept would be as a condition on `V : ι → Submodule 𝕜 E`.  We
instead implement it as a condition on a family of inner product spaces each equipped with an
isometric embedding into `E`, thus making it a property of morphisms rather than subobjects.
The connection to the subobject spelling is shown in `orthogonalFamily_iff_pairwise`.

This definition is less lightweight, but allows for better definitional properties when the inner
product space structure on each of the submodules is important -- for example, when considering
their Hilbert sum (`PiLp V 2`).  For example, given an orthonormal set of vectors `v : ι → E`,
we have an associated orthogonal family of one-dimensional subspaces of `E`, which it is convenient
to be able to discuss using `ι → 𝕜` rather than `Π i : ι, span 𝕜 (v i)`. -/
def OrthogonalFamily (G : ι → Type*) [∀ i, SeminormedAddCommGroup (G i)]
    [∀ i, InnerProductSpace 𝕜 (G i)] (V : ∀ i, G i →ₗᵢ[𝕜] E) : Prop :=
  Pairwise fun i j => ∀ v : G i, ∀ w : G j, ⟪V i v, V j w⟫ = 0


theorem Orthonormal.orthogonalFamily {v : ι → E} (hv : Orthonormal 𝕜 v) :
    OrthogonalFamily 𝕜 (fun _i : ι => 𝕜) fun i => LinearIsometry.toSpanSingleton 𝕜 E (hv.1 i) :=
                        /-
                          𝕜 : Type u_1
                          E : Type u_2
                          inst✝² : RCLike 𝕜
                          inst✝¹ : SeminormedAddCommGroup E
                          inst✝ : InnerProductSpace 𝕜 E
                          ι : Type u_4
                          v : ι → E
                          hv : Orthonormal 𝕜 v
                          i j : ι
                          hij : Ne i j
                          a : (fun _i => 𝕜) i
                          b : (fun _i => 𝕜) j
                          ⊢ Eq (Inner.inner (((fun i => LinearIsometry.toSpanSingleton 𝕜 E ⋯) i) a) (((f …
                        -/
  fun i j hij a b => by simp [inner_smul_left, inner_smul_right, hv.2 hij]
                        /-
                          🎉 no goals
                        -/


theorem OrthogonalFamily.eq_ite [DecidableEq ι] {i j : ι} (v : G i) (w : G j) :
    ⟪V i v, V j w⟫ = ite (i = j) ⟪V i v, V j w⟫ 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_4
    G : ι → Type u_5
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    inst✝ : DecidableEq ι
    i j : ι
    v : G i
    w : G j
    ⊢ Eq (Inner.inner ((V i) v) ((V j) w)) (ite (Eq i j) (Inner.inner ((V i) v) (( …
  -/
  split_ifs with h
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : SeminormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝² : (i : ι) → NormedAddCommGroup (G i)
      inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      inst✝ : DecidableEq ι
      i j : ι
      v : G i
      w : G j
      h : Eq i j
      ⊢ Eq (Inner.inner ((V i) v) ((V j) w)) (Inner.inner ((V i) v) ((V j) w))
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : SeminormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝² : (i : ι) → NormedAddCommGroup (G i)
      inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      inst✝ : DecidableEq ι
      i j : ι
      v : G i
      w : G j
      h : Not (Eq i j)
      ⊢ Eq (Inner.inner ((V i) v) ((V j) w)) 0
    -/
  · exact hV h v w
    /-
      🎉 no goals
    -/


theorem OrthogonalFamily.inner_right_dfinsupp
    [∀ (i) (x : G i), Decidable (x ≠ 0)] [DecidableEq ι] (l : ⨁ i, G i) (i : ι) (v : G i) :
    ⟪V i v, l.sum fun j => V j⟫ = ⟪v, l i⟫ :=
  calc
    ⟪V i v, l.sum fun j => V j⟫ = l.sum fun j => fun w => ⟪V i v, V j w⟫ :=
      DFinsupp.inner_sum (fun j => V j) l (V i v)
    _ = l.sum fun j => fun w => ite (i = j) ⟪V i v, V j w⟫ 0 :=
      (congr_arg l.sum <| funext fun _ => funext <| hV.eq_ite v)
    _ = ⟪v, l i⟫ := by
      simp only [DFinsupp.sum, Submodule.coe_inner, Finset.sum_ite_eq, ite_eq_left_iff,
        DFinsupp.mem_support_toFun]
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : SeminormedAddCommGroup E
        inst✝⁴ : InnerProductSpace 𝕜 E
        ι : Type u_4
        G : ι → Type u_5
        inst✝³ : (i : ι) → NormedAddCommGroup (G i)
        inst✝² : (i : ι) → InnerProductSpace 𝕜 (G i)
        V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
        hV : OrthogonalFamily 𝕜 G V
        inst✝¹ : (i : ι) → (x : G i) → Decidable (Ne x 0)
        inst✝ : DecidableEq ι
        l : DirectSum ι fun i => G i
        i : ι
        v : G i
        ⊢ Eq (ite (Ne (l i) 0) (Inner.inner ((V i) v) ((V i) (l i))) 0) (Inner.inner v …
      -/
      split_ifs with h
        /-
          case pos
          𝕜 : Type u_1
          E : Type u_2
          inst✝⁶ : RCLike 𝕜
          inst✝⁵ : SeminormedAddCommGroup E
          inst✝⁴ : InnerProductSpace 𝕜 E
          ι : Type u_4
          G : ι → Type u_5
          inst✝³ : (i : ι) → NormedAddCommGroup (G i)
          inst✝² : (i : ι) → InnerProductSpace 𝕜 (G i)
          V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
          hV : OrthogonalFamily 𝕜 G V
          inst✝¹ : (i : ι) → (x : G i) → Decidable (Ne x 0)
          inst✝ : DecidableEq ι
          l : DirectSum ι fun i => G i
          i : ι
          v : G i
          h : Ne (l i) 0
          ⊢ Eq (Inner.inner ((V i) v) ((V i) (l i))) (Inner.inner v (l i))
        -/
      · simp only [LinearIsometry.inner_map_map]
        /-
          🎉 no goals
        -/
        /-
          case neg
          𝕜 : Type u_1
          E : Type u_2
          inst✝⁶ : RCLike 𝕜
          inst✝⁵ : SeminormedAddCommGroup E
          inst✝⁴ : InnerProductSpace 𝕜 E
          ι : Type u_4
          G : ι → Type u_5
          inst✝³ : (i : ι) → NormedAddCommGroup (G i)
          inst✝² : (i : ι) → InnerProductSpace 𝕜 (G i)
          V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
          hV : OrthogonalFamily 𝕜 G V
          inst✝¹ : (i : ι) → (x : G i) → Decidable (Ne x 0)
          inst✝ : DecidableEq ι
          l : DirectSum ι fun i => G i
          i : ι
          v : G i
          h : Not (Ne (l i) 0)
          ⊢ Eq 0 (Inner.inner v (l i))
        -/
      · simp only [of_not_not h, inner_zero_right]
        /-
          🎉 no goals
        -/


theorem OrthogonalFamily.inner_right_fintype [Fintype ι] (l : ∀ i, G i) (i : ι) (v : G i) :
    ⟪V i v, ∑ j : ι, V j (l j)⟫ = ⟪v, l i⟫ := by
  classical
  calc
    ⟪V i v, ∑ j : ι, V j (l j)⟫ = ∑ j : ι, ⟪V i v, V j (l j)⟫ := by rw [inner_sum]
    _ = ∑ j, ite (i = j) ⟪V i v, V j (l j)⟫ 0 :=
      (congr_arg (Finset.sum Finset.univ) <| funext fun j => hV.eq_ite v (l j))
    _ = ⟪v, l i⟫ := by
      simp only [Finset.sum_ite_eq, Finset.mem_univ, (V i).inner_map_map, if_true]


nonrec theorem OrthogonalFamily.inner_sum (l₁ l₂ : ∀ i, G i) (s : Finset ι) :
    ⟪∑ i ∈ s, V i (l₁ i), ∑ j ∈ s, V j (l₂ j)⟫ = ∑ i ∈ s, ⟪l₁ i, l₂ i⟫ := by
  classical
  calc
    ⟪∑ i ∈ s, V i (l₁ i), ∑ j ∈ s, V j (l₂ j)⟫ = ∑ j ∈ s, ∑ i ∈ s, ⟪V i (l₁ i), V j (l₂ j)⟫ := by
      simp only [sum_inner, inner_sum]
    _ = ∑ j ∈ s, ∑ i ∈ s, ite (i = j) ⟪V i (l₁ i), V j (l₂ j)⟫ 0 := by
      congr with i
      congr with j
      apply hV.eq_ite
    _ = ∑ i ∈ s, ⟪l₁ i, l₂ i⟫ := by
      simp only [Finset.sum_ite_of_true, Finset.sum_ite_eq', LinearIsometry.inner_map_map,
        imp_self, imp_true_iff]


theorem OrthogonalFamily.norm_sum (l : ∀ i, G i) (s : Finset ι) :
    ‖∑ i ∈ s, V i (l i)‖ ^ 2 = ∑ i ∈ s, ‖l i‖ ^ 2 := by
  have : ((‖∑ i ∈ s, V i (l i)‖ : ℝ) : 𝕜) ^ 2 = ∑ i ∈ s, ((‖l i‖ : ℝ) : 𝕜) ^ 2 := by
    simp only [← inner_self_eq_norm_sq_to_K, hV.inner_sum]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    ι : Type u_4
    G : ι → Type u_5
    inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    l : (i : ι) → G i
    s : Finset ι
    this : Eq (HPow.hPow (↑(Norm.norm (s.sum fun i => (V i) (l i)))) 2) (s.sum fun …
    ⊢ Eq (HPow.hPow (Norm.norm (s.sum fun i => (V i) (l i))) 2) (s.sum fun i => HP …
  -/
  exact mod_cast this
  /-
    🎉 no goals
  -/


/-- The composition of an orthogonal family of subspaces with an injective function is also an
orthogonal family. -/
theorem OrthogonalFamily.comp {γ : Type*} {f : γ → ι} (hf : Function.Injective f) :
    OrthogonalFamily 𝕜 (fun g => G (f g)) fun g => V (f g) :=
  fun _i _j hij v w => hV (hf.ne hij) v w


theorem OrthogonalFamily.orthonormal_sigma_orthonormal {α : ι → Type*} {v_family : ∀ i, α i → G i}
    (hv_family : ∀ i, Orthonormal 𝕜 (v_family i)) :
    Orthonormal 𝕜 fun a : Σi, α i => V a.1 (v_family a.1 a.2) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    ι : Type u_4
    G : ι → Type u_5
    inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    α : ι → Type u_6
    v_family : (i : ι) → α i → G i
    hv_family : ∀ (i : ι), Orthonormal 𝕜 (v_family i)
    ⊢ Orthonormal 𝕜 fun a => (V a.fst) (v_family a.fst a.snd)
  -/
  constructor
    /-
      case left
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : RCLike 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
      inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      α : ι → Type u_6
      v_family : (i : ι) → α i → G i
      hv_family : ∀ (i : ι), Orthonormal 𝕜 (v_family i)
      ⊢ ∀ (i : Sigma fun i => α i), Eq (Norm.norm ((fun a => (V a.fst) (v_family a.f …
    -/
  · rintro ⟨i, v⟩
    /-
      case left.mk
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : RCLike 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
      inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      α : ι → Type u_6
      v_family : (i : ι) → α i → G i
      hv_family : ∀ (i : ι), Orthonormal 𝕜 (v_family i)
      i : ι
      v : α i
      ⊢ Eq (Norm.norm ((fun a => (V a.fst) (v_family a.fst a.snd)) ⟨i, v⟩)) 1
    -/
    simpa only [LinearIsometry.norm_map] using (hv_family i).left v
    /-
      🎉 no goals
    -/
  /-
    case right
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    ι : Type u_4
    G : ι → Type u_5
    inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    α : ι → Type u_6
    v_family : (i : ι) → α i → G i
    hv_family : ∀ (i : ι), Orthonormal 𝕜 (v_family i)
    ⊢ Pairwise fun i j => Eq (Inner.inner ((fun a => (V a.fst) (v_family a.fst a.s …
  -/
  rintro ⟨i, v⟩ ⟨j, w⟩ hvw
  /-
    case right.mk.mk
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    ι : Type u_4
    G : ι → Type u_5
    inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    α : ι → Type u_6
    v_family : (i : ι) → α i → G i
    hv_family : ∀ (i : ι), Orthonormal 𝕜 (v_family i)
    i : ι
    v : α i
    j : ι
    w : α j
    hvw : Ne ⟨i, v⟩ ⟨j, w⟩
    ⊢ Eq (Inner.inner ((fun a => (V a.fst) (v_family a.fst a.snd)) ⟨i, v⟩) ((fun a …
  -/
  by_cases hij : i = j
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : RCLike 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
      inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      α : ι → Type u_6
      v_family : (i : ι) → α i → G i
      hv_family : ∀ (i : ι), Orthonormal 𝕜 (v_family i)
      i : ι
      v : α i
      j : ι
      w : α j
      hvw : Ne ⟨i, v⟩ ⟨j, w⟩
      hij : Eq i j
      ⊢ Eq (Inner.inner ((fun a => (V a.fst) (v_family a.fst a.snd)) ⟨i, v⟩) ((fun a …
    -/
  · subst hij
    have : v ≠ w := fun h => by
      subst h
      exact hvw rfl
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : RCLike 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
      inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      α : ι → Type u_6
      v_family : (i : ι) → α i → G i
      hv_family : ∀ (i : ι), Orthonormal 𝕜 (v_family i)
      i : ι
      v w : α i
      hvw : Ne ⟨i, v⟩ ⟨i, w⟩
      this : Ne v w
      ⊢ Eq (Inner.inner ((fun a => (V a.fst) (v_family a.fst a.snd)) ⟨i, v⟩) ((fun a …
    -/
    simpa only [LinearIsometry.inner_map_map] using (hv_family i).2 this
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : RCLike 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
      inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      α : ι → Type u_6
      v_family : (i : ι) → α i → G i
      hv_family : ∀ (i : ι), Orthonormal 𝕜 (v_family i)
      i : ι
      v : α i
      j : ι
      w : α j
      hvw : Ne ⟨i, v⟩ ⟨j, w⟩
      hij : Not (Eq i j)
      ⊢ Eq (Inner.inner ((fun a => (V a.fst) (v_family a.fst a.snd)) ⟨i, v⟩) ((fun a …
    -/
  · exact hV hij (v_family i v) (v_family j w)
    /-
      🎉 no goals
    -/


theorem OrthogonalFamily.norm_sq_diff_sum [DecidableEq ι] (f : ∀ i, G i) (s₁ s₂ : Finset ι) :
    ‖(∑ i ∈ s₁, V i (f i)) - ∑ i ∈ s₂, V i (f i)‖ ^ 2 =
      (∑ i ∈ s₁ \ s₂, ‖f i‖ ^ 2) + ∑ i ∈ s₂ \ s₁, ‖f i‖ ^ 2 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_4
    G : ι → Type u_5
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    inst✝ : DecidableEq ι
    f : (i : ι) → G i
    s₁ s₂ : Finset ι
    ⊢ Eq (HPow.hPow (Norm.norm (HSub.hSub (s₁.sum fun i => (V i) (f i)) (s₂.sum fu …
  -/
  rw [← Finset.sum_sdiff_sub_sum_sdiff, sub_eq_add_neg, ← Finset.sum_neg_distrib]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_4
    G : ι → Type u_5
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    inst✝ : DecidableEq ι
    f : (i : ι) → G i
    s₁ s₂ : Finset ι
    ⊢ Eq (HPow.hPow (Norm.norm (HAdd.hAdd ((SDiff.sdiff s₁ s₂).sum fun x => (V x)  …
  -/
  let F : ∀ i, G i := fun i => if i ∈ s₁ then f i else -f i
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_4
    G : ι → Type u_5
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    inst✝ : DecidableEq ι
    f : (i : ι) → G i
    s₁ s₂ : Finset ι
    F : (i : ι) → G i := fun i => ite (Membership.mem s₁ i) (f i) (Neg.neg (f i))
    ⊢ Eq (HPow.hPow (Norm.norm (HAdd.hAdd ((SDiff.sdiff s₁ s₂).sum fun x => (V x)  …
  -/
  have hF₁ : ∀ i ∈ s₁ \ s₂, F i = f i := fun i hi => if_pos (Finset.sdiff_subset hi)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_4
    G : ι → Type u_5
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    inst✝ : DecidableEq ι
    f : (i : ι) → G i
    s₁ s₂ : Finset ι
    F : (i : ι) → G i := fun i => ite (Membership.mem s₁ i) (f i) (Neg.neg (f i))
    hF₁ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₁ s₂) i → Eq (F i) (f i)
    ⊢ Eq (HPow.hPow (Norm.norm (HAdd.hAdd ((SDiff.sdiff s₁ s₂).sum fun x => (V x)  …
  -/
  have hF₂ : ∀ i ∈ s₂ \ s₁, F i = -f i := fun i hi => if_neg (Finset.mem_sdiff.mp hi).2
  have hF : ∀ i, ‖F i‖ = ‖f i‖ := by
    intro i
    dsimp only [F]
    split_ifs <;> simp only [eq_self_iff_true, norm_neg]
  have :
    ‖(∑ i ∈ s₁ \ s₂, V i (F i)) + ∑ i ∈ s₂ \ s₁, V i (F i)‖ ^ 2 =
      (∑ i ∈ s₁ \ s₂, ‖F i‖ ^ 2) + ∑ i ∈ s₂ \ s₁, ‖F i‖ ^ 2 := by
    have hs : Disjoint (s₁ \ s₂) (s₂ \ s₁) := disjoint_sdiff_sdiff
    simpa only [Finset.sum_union hs] using hV.norm_sum F (s₁ \ s₂ ∪ s₂ \ s₁)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_4
    G : ι → Type u_5
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    inst✝ : DecidableEq ι
    f : (i : ι) → G i
    s₁ s₂ : Finset ι
    F : (i : ι) → G i := fun i => ite (Membership.mem s₁ i) (f i) (Neg.neg (f i))
    hF₁ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₁ s₂) i → Eq (F i) (f i)
    hF₂ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₂ s₁) i → Eq (F i) (Neg.neg (f i))
    hF : ∀ (i : ι), Eq (Norm.norm (F i)) (Norm.norm (f i))
    this : Eq (HPow.hPow (Norm.norm (HAdd.hAdd ((SDiff.sdiff s₁ s₂).sum fun i => ( …
    ⊢ Eq (HPow.hPow (Norm.norm (HAdd.hAdd ((SDiff.sdiff s₁ s₂).sum fun x => (V x)  …
  -/
  convert this using 4
    /-
      case h.e'_2.h.e'_5.h.e'_3.h.e'_5
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : SeminormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝² : (i : ι) → NormedAddCommGroup (G i)
      inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      inst✝ : DecidableEq ι
      f : (i : ι) → G i
      s₁ s₂ : Finset ι
      F : (i : ι) → G i := fun i => ite (Membership.mem s₁ i) (f i) (Neg.neg (f i))
      hF₁ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₁ s₂) i → Eq (F i) (f i)
      hF₂ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₂ s₁) i → Eq (F i) (Neg.neg (f i))
      hF : ∀ (i : ι), Eq (Norm.norm (F i)) (Norm.norm (f i))
      this : Eq (HPow.hPow (Norm.norm (HAdd.hAdd ((SDiff.sdiff s₁ s₂).sum fun i => ( …
      ⊢ Eq ((SDiff.sdiff s₁ s₂).sum fun x => (V x) (f x)) ((SDiff.sdiff s₁ s₂).sum f …
    -/
  · refine Finset.sum_congr rfl fun i hi => ?_
    /-
      case h.e'_2.h.e'_5.h.e'_3.h.e'_5
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : SeminormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝² : (i : ι) → NormedAddCommGroup (G i)
      inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      inst✝ : DecidableEq ι
      f : (i : ι) → G i
      s₁ s₂ : Finset ι
      F : (i : ι) → G i := fun i => ite (Membership.mem s₁ i) (f i) (Neg.neg (f i))
      hF₁ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₁ s₂) i → Eq (F i) (f i)
      hF₂ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₂ s₁) i → Eq (F i) (Neg.neg (f i))
      hF : ∀ (i : ι), Eq (Norm.norm (F i)) (Norm.norm (f i))
      this : Eq (HPow.hPow (Norm.norm (HAdd.hAdd ((SDiff.sdiff s₁ s₂).sum fun i => ( …
      i : ι
      hi : Membership.mem (SDiff.sdiff s₁ s₂) i
      ⊢ Eq ((V i) (f i)) ((V i) (F i))
    -/
    simp only [hF₁ i hi]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_5.h.e'_3.h.e'_6
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : SeminormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝² : (i : ι) → NormedAddCommGroup (G i)
      inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      inst✝ : DecidableEq ι
      f : (i : ι) → G i
      s₁ s₂ : Finset ι
      F : (i : ι) → G i := fun i => ite (Membership.mem s₁ i) (f i) (Neg.neg (f i))
      hF₁ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₁ s₂) i → Eq (F i) (f i)
      hF₂ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₂ s₁) i → Eq (F i) (Neg.neg (f i))
      hF : ∀ (i : ι), Eq (Norm.norm (F i)) (Norm.norm (f i))
      this : Eq (HPow.hPow (Norm.norm (HAdd.hAdd ((SDiff.sdiff s₁ s₂).sum fun i => ( …
      ⊢ Eq ((SDiff.sdiff s₂ s₁).sum fun x => Neg.neg ((V x) (f x))) ((SDiff.sdiff s₂ …
    -/
  · refine Finset.sum_congr rfl fun i hi => ?_
    /-
      case h.e'_2.h.e'_5.h.e'_3.h.e'_6
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : SeminormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝² : (i : ι) → NormedAddCommGroup (G i)
      inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      inst✝ : DecidableEq ι
      f : (i : ι) → G i
      s₁ s₂ : Finset ι
      F : (i : ι) → G i := fun i => ite (Membership.mem s₁ i) (f i) (Neg.neg (f i))
      hF₁ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₁ s₂) i → Eq (F i) (f i)
      hF₂ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₂ s₁) i → Eq (F i) (Neg.neg (f i))
      hF : ∀ (i : ι), Eq (Norm.norm (F i)) (Norm.norm (f i))
      this : Eq (HPow.hPow (Norm.norm (HAdd.hAdd ((SDiff.sdiff s₁ s₂).sum fun i => ( …
      i : ι
      hi : Membership.mem (SDiff.sdiff s₂ s₁) i
      ⊢ Eq (Neg.neg ((V i) (f i))) ((V i) (F i))
    -/
    simp only [hF₂ i hi, LinearIsometry.map_neg]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_5.a.h.e'_5
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : SeminormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝² : (i : ι) → NormedAddCommGroup (G i)
      inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      inst✝ : DecidableEq ι
      f : (i : ι) → G i
      s₁ s₂ : Finset ι
      F : (i : ι) → G i := fun i => ite (Membership.mem s₁ i) (f i) (Neg.neg (f i))
      hF₁ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₁ s₂) i → Eq (F i) (f i)
      hF₂ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₂ s₁) i → Eq (F i) (Neg.neg (f i))
      hF : ∀ (i : ι), Eq (Norm.norm (F i)) (Norm.norm (f i))
      this : Eq (HPow.hPow (Norm.norm (HAdd.hAdd ((SDiff.sdiff s₁ s₂).sum fun i => ( …
      x✝ : ι
      a✝ : Membership.mem (SDiff.sdiff s₁ s₂) x✝
      ⊢ Eq (Norm.norm (f x✝)) (Norm.norm (F x✝))
    -/
  · simp only [hF]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_6.a.h.e'_5
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : SeminormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_4
      G : ι → Type u_5
      inst✝² : (i : ι) → NormedAddCommGroup (G i)
      inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      inst✝ : DecidableEq ι
      f : (i : ι) → G i
      s₁ s₂ : Finset ι
      F : (i : ι) → G i := fun i => ite (Membership.mem s₁ i) (f i) (Neg.neg (f i))
      hF₁ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₁ s₂) i → Eq (F i) (f i)
      hF₂ : ∀ (i : ι), Membership.mem (SDiff.sdiff s₂ s₁) i → Eq (F i) (Neg.neg (f i))
      hF : ∀ (i : ι), Eq (Norm.norm (F i)) (Norm.norm (f i))
      this : Eq (HPow.hPow (Norm.norm (HAdd.hAdd ((SDiff.sdiff s₁ s₂).sum fun i => ( …
      x✝ : ι
      a✝ : Membership.mem (SDiff.sdiff s₂ s₁) x✝
      ⊢ Eq (Norm.norm (f x✝)) (Norm.norm (F x✝))
    -/
  · simp only [hF]
    /-
      🎉 no goals
    -/


/-- A family `f` of mutually-orthogonal elements of `E` is summable, if and only if
`(fun i ↦ ‖f i‖ ^ 2)` is summable. -/
theorem OrthogonalFamily.summable_iff_norm_sq_summable [CompleteSpace E] (f : ∀ i, G i) :
    (Summable fun i => V i (f i)) ↔ Summable fun i => ‖f i‖ ^ 2 := by
  classical
    simp only [summable_iff_cauchySeq_finset, NormedAddCommGroup.cauchySeq_iff, Real.norm_eq_abs]
    constructor
    · intro hf ε hε
      obtain ⟨a, H⟩ := hf _ (sqrt_pos.mpr hε)
      use a
      intro s₁ hs₁ s₂ hs₂
      rw [← Finset.sum_sdiff_sub_sum_sdiff]
      refine (abs_sub _ _).trans_lt ?_
      have : ∀ i, 0 ≤ ‖f i‖ ^ 2 := fun i : ι => sq_nonneg _
      simp only [Finset.abs_sum_of_nonneg' this]
      have : ((∑ i ∈ s₁ \ s₂, ‖f i‖ ^ 2) + ∑ i ∈ s₂ \ s₁, ‖f i‖ ^ 2) < √ε ^ 2 := by
        rw [← hV.norm_sq_diff_sum, sq_lt_sq, abs_of_nonneg (sqrt_nonneg _),
          abs_of_nonneg (norm_nonneg _)]
        exact H s₁ hs₁ s₂ hs₂
      have hη := sq_sqrt (le_of_lt hε)
      linarith
    · intro hf ε hε
      have hε' : 0 < ε ^ 2 / 2 := half_pos (sq_pos_of_pos hε)
      obtain ⟨a, H⟩ := hf _ hε'
      use a
      intro s₁ hs₁ s₂ hs₂
      refine (abs_lt_of_sq_lt_sq' ?_ (le_of_lt hε)).2
      have has : a ≤ s₁ ⊓ s₂ := le_inf hs₁ hs₂
      rw [hV.norm_sq_diff_sum]
      have Hs₁ : ∑ x ∈ s₁ \ s₂, ‖f x‖ ^ 2 < ε ^ 2 / 2 := by
        convert H _ hs₁ _ has
        have : s₁ ⊓ s₂ ⊆ s₁ := Finset.inter_subset_left
        rw [← Finset.sum_sdiff this, add_tsub_cancel_right, Finset.abs_sum_of_nonneg']
        · simp
        · exact fun i => sq_nonneg _
      have Hs₂ : ∑ x ∈ s₂ \ s₁, ‖f x‖ ^ 2 < ε ^ 2 / 2 := by
        convert H _ hs₂ _ has
        have : s₁ ⊓ s₂ ⊆ s₂ := Finset.inter_subset_right
        rw [← Finset.sum_sdiff this, add_tsub_cancel_right, Finset.abs_sum_of_nonneg']
        · simp
        · exact fun i => sq_nonneg _
      linarith


/-- An orthogonal family forms an independent family of subspaces; that is, any collection of
elements each from a different subspace in the family is linearly independent. In particular, the
pairwise intersections of elements of the family are 0. -/
theorem OrthogonalFamily.independent {V : ι → Submodule 𝕜 E}
    (hV : OrthogonalFamily 𝕜 (fun i => V i) fun i => (V i).subtypeₗᵢ) :
    iSupIndep V := by
  classical
  apply iSupIndep_of_dfinsupp_lsum_injective
  refine LinearMap.ker_eq_bot.mp ?_
  rw [Submodule.eq_bot_iff]
  intro v hv
  rw [LinearMap.mem_ker] at hv
  ext i
  suffices ⟪(v i : E), v i⟫ = 0 by simpa only [inner_self_eq_zero] using this
  calc
    ⟪(v i : E), v i⟫ = ⟪(v i : E), DFinsupp.lsum ℕ (fun i => (V i).subtype) v⟫ := by
      simpa only [DFinsupp.sumAddHom_apply, DFinsupp.lsum_apply_apply] using
        (hV.inner_right_dfinsupp v i (v i)).symm
    _ = 0 := by simp only [hv, inner_zero_right]


theorem DirectSum.IsInternal.collectedBasis_orthonormal [DecidableEq ι] {V : ι → Submodule 𝕜 E}
    (hV : OrthogonalFamily 𝕜 (fun i => V i) fun i => (V i).subtypeₗᵢ)
    (hV_sum : DirectSum.IsInternal fun i => V i) {α : ι → Type*}
    {v_family : ∀ i, Basis (α i) 𝕜 (V i)} (hv_family : ∀ i, Orthonormal 𝕜 (v_family i)) :
    Orthonormal 𝕜 (hV_sum.collectedBasis v_family) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    ι : Type u_4
    inst✝ : DecidableEq ι
    V : ι → Submodule 𝕜 E
    hV : OrthogonalFamily 𝕜 (fun i => Subtype fun x => Membership.mem (V i) x) fun …
    hV_sum : DirectSum.IsInternal fun i => V i
    α : ι → Type u_6
    v_family : (i : ι) → Basis (α i) 𝕜 (Subtype fun x => Membership.mem (V i) x)
    hv_family : ∀ (i : ι), Orthonormal 𝕜 ⇑(v_family i)
    ⊢ Orthonormal 𝕜 ⇑(hV_sum.collectedBasis v_family)
  -/
  simpa only [hV_sum.collectedBasis_coe] using hV.orthonormal_sigma_orthonormal hv_family
  /-
    🎉 no goals
  -/


