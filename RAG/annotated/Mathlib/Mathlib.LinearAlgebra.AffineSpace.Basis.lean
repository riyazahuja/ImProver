/-- An affine basis is a family of affine-independent points whose span is the top subspace. -/
structure AffineBasis (ι : Type u₁) (k : Type u₂) {V : Type u₃} (P : Type u₄) [AddCommGroup V]
  [AffineSpace V P] [Ring k] [Module k V] where
  protected toFun : ι → P
  protected ind' : AffineIndependent k toFun
  protected tot' : affineSpan k (range toFun) = ⊤


/-- The unique point in a single-point space is the simplest example of an affine basis. -/
instance : Inhabited (AffineBasis PUnit k PUnit) :=
                                                   /-
                                                     ι : Type u_1
                                                     ι' : Type u_2
                                                     G : Type u_3
                                                     G' : Type u_4
                                                     k : Type u_5
                                                     V : Type u_6
                                                     P : Type u_7
                                                     inst✝³ : AddCommGroup V
                                                     inst✝² : AddTorsor V P
                                                     inst✝¹ : Ring k
                                                     inst✝ : Module k V
                                                     b : AffineBasis ι k P
                                                     s : Finset ι
                                                     i j : ι
                                                     e : Equiv ι ι'
                                                     ⊢ Eq (affineSpan k (Set.range id)) Top.top
                                                   -/
  ⟨⟨id, affineIndependent_of_subsingleton k id, by simp⟩⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


instance instFunLike : FunLike (AffineBasis ι k P) ι P where
  coe := AffineBasis.toFun
                             /-
                               ι : Type u_1
                               ι' : Type u_2
                               G : Type u_3
                               G' : Type u_4
                               k : Type u_5
                               V : Type u_6
                               P : Type u_7
                               inst✝³ : AddCommGroup V
                               inst✝² : AddTorsor V P
                               inst✝¹ : Ring k
                               inst✝ : Module k V
                               b : AffineBasis ι k P
                               s : Finset ι
                               i j : ι
                               e : Equiv ι ι'
                               f g : AffineBasis ι k P
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


@[ext]
theorem ext {b₁ b₂ : AffineBasis ι k P} (h : (b₁ : ι → P) = b₂) : b₁ = b₂ :=
  DFunLike.coe_injective h


theorem ind : AffineIndependent k b :=
  b.ind'


theorem tot : affineSpan k (range b) = ⊤ :=
  b.tot'


include b in
protected theorem nonempty : Nonempty ι :=
  not_isEmpty_iff.mp fun hι => by
    /-
      ι : Type u_1
      k : Type u_5
      V : Type u_6
      P : Type u_7
      inst✝³ : AddCommGroup V
      inst✝² : AddTorsor V P
      inst✝¹ : Ring k
      inst✝ : Module k V
      b : AffineBasis ι k P
      hι : IsEmpty ι
      ⊢ False
    -/
    simpa only [@range_eq_empty _ _ hι, AffineSubspace.span_empty, bot_ne_top] using b.tot
    /-
      🎉 no goals
    -/


/-- Composition of an affine basis and an equivalence of index types. -/
def reindex (e : ι ≃ ι') : AffineBasis ι' k P :=
  ⟨b ∘ e.symm, b.ind.comp_embedding e.symm.toEmbedding, by
    /-
      ι : Type u_1
      ι' : Type u_2
      G : Type u_3
      G' : Type u_4
      k : Type u_5
      V : Type u_6
      P : Type u_7
      inst✝³ : AddCommGroup V
      inst✝² : AddTorsor V P
      inst✝¹ : Ring k
      inst✝ : Module k V
      b : AffineBasis ι k P
      s : Finset ι
      i j : ι
      e✝ e : Equiv ι ι'
      ⊢ Eq (affineSpan k (Set.range (Function.comp ⇑b ⇑e.symm))) Top.top
    -/
    rw [e.symm.surjective.range_comp]
    /-
      ι : Type u_1
      ι' : Type u_2
      G : Type u_3
      G' : Type u_4
      k : Type u_5
      V : Type u_6
      P : Type u_7
      inst✝³ : AddCommGroup V
      inst✝² : AddTorsor V P
      inst✝¹ : Ring k
      inst✝ : Module k V
      b : AffineBasis ι k P
      s : Finset ι
      i j : ι
      e✝ e : Equiv ι ι'
      ⊢ Eq (affineSpan k (Set.range ⇑b)) Top.top
    -/
    exact b.3⟩
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem coe_reindex : ⇑(b.reindex e) = b ∘ e.symm :=
  rfl


@[simp]
theorem reindex_apply (i' : ι') : b.reindex e i' = b (e.symm i') :=
  rfl


@[simp]
theorem reindex_refl : b.reindex (Equiv.refl _) = b :=
  ext rfl


/-- Given an affine basis for an affine space `P`, if we single out one member of the family, we
obtain a linear basis for the model space `V`.

The linear basis corresponding to the singled-out member `i : ι` is indexed by `{j : ι // j ≠ i}`
and its `j`th element is `b j -ᵥ b i`. (See `basisOf_apply`.) -/
noncomputable def basisOf (i : ι) : Basis { j : ι // j ≠ i } k V :=
  Basis.mk ((affineIndependent_iff_linearIndependent_vsub k b i).mp b.ind)
    (by
      suffices
        Submodule.span k (range fun j : { x // x ≠ i } => b ↑j -ᵥ b i) = vectorSpan k (range b) by
        rw [this, ← direction_affineSpan, b.tot, AffineSubspace.direction_top]
      /-
        ι : Type u_1
        ι' : Type u_2
        G : Type u_3
        G' : Type u_4
        k : Type u_5
        V : Type u_6
        P : Type u_7
        inst✝³ : AddCommGroup V
        inst✝² : AddTorsor V P
        inst✝¹ : Ring k
        inst✝ : Module k V
        b : AffineBasis ι k P
        s : Finset ι
        i✝ j : ι
        e : Equiv ι ι'
        i : ι
        ⊢ Eq (Submodule.span k (Set.range fun j => VSub.vsub (b ↑j) (b i))) (vectorSpa …
      -/
      conv_rhs => rw [← image_univ]
      /-
        ι : Type u_1
        ι' : Type u_2
        G : Type u_3
        G' : Type u_4
        k : Type u_5
        V : Type u_6
        P : Type u_7
        inst✝³ : AddCommGroup V
        inst✝² : AddTorsor V P
        inst✝¹ : Ring k
        inst✝ : Module k V
        b : AffineBasis ι k P
        s : Finset ι
        i✝ j : ι
        e : Equiv ι ι'
        i : ι
        ⊢ Eq (Submodule.span k (Set.range fun j => VSub.vsub (b ↑j) (b i))) (vectorSpa …
      -/
      rw [vectorSpan_image_eq_span_vsub_set_right_ne k b (mem_univ i)]
      /-
        ι : Type u_1
        ι' : Type u_2
        G : Type u_3
        G' : Type u_4
        k : Type u_5
        V : Type u_6
        P : Type u_7
        inst✝³ : AddCommGroup V
        inst✝² : AddTorsor V P
        inst✝¹ : Ring k
        inst✝ : Module k V
        b : AffineBasis ι k P
        s : Finset ι
        i✝ j : ι
        e : Equiv ι ι'
        i : ι
        ⊢ Eq (Submodule.span k (Set.range fun j => VSub.vsub (b ↑j) (b i))) (Submodule …
      -/
      congr
      /-
        case e_s
        ι : Type u_1
        ι' : Type u_2
        G : Type u_3
        G' : Type u_4
        k : Type u_5
        V : Type u_6
        P : Type u_7
        inst✝³ : AddCommGroup V
        inst✝² : AddTorsor V P
        inst✝¹ : Ring k
        inst✝ : Module k V
        b : AffineBasis ι k P
        s : Finset ι
        i✝ j : ι
        e : Equiv ι ι'
        i : ι
        ⊢ Eq (Set.range fun j => VSub.vsub (b ↑j) (b i)) (Set.image (fun x => VSub.vsu …
      -/
      ext v
      /-
        case e_s.h
        ι : Type u_1
        ι' : Type u_2
        G : Type u_3
        G' : Type u_4
        k : Type u_5
        V : Type u_6
        P : Type u_7
        inst✝³ : AddCommGroup V
        inst✝² : AddTorsor V P
        inst✝¹ : Ring k
        inst✝ : Module k V
        b : AffineBasis ι k P
        s : Finset ι
        i✝ j : ι
        e : Equiv ι ι'
        i : ι
        v : V
        ⊢ Iff (Membership.mem (Set.range fun j => VSub.vsub (b ↑j) (b i)) v) (Membersh …
      -/
      simp)
      /-
        🎉 no goals
      -/


@[simp]
theorem basisOf_apply (i : ι) (j : { j : ι // j ≠ i }) : b.basisOf i j = b ↑j -ᵥ b i := by
  /-
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : Ring k
    inst✝ : Module k V
    b : AffineBasis ι k P
    i : ι
    j : Subtype fun j => Ne j i
    ⊢ Eq ((b.basisOf i) j) (VSub.vsub (b ↑j) (b i))
  -/
  simp [basisOf]
  /-
    🎉 no goals
  -/


@[simp]
theorem basisOf_reindex (i : ι') :
    (b.reindex e).basisOf i =
      (b.basisOf <| e.symm i).reindex (e.subtypeEquiv fun _ => e.eq_symm_apply.not) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : Ring k
    inst✝ : Module k V
    b : AffineBasis ι k P
    e : Equiv ι ι'
    i : ι'
    ⊢ Eq ((b.reindex e).basisOf i) ((b.basisOf (e.symm i)).reindex (e.subtypeEquiv …
  -/
  ext j
  /-
    case a
    ι : Type u_1
    ι' : Type u_2
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : Ring k
    inst✝ : Module k V
    b : AffineBasis ι k P
    e : Equiv ι ι'
    i : ι'
    j : Subtype fun j => Ne j i
    ⊢ Eq (((b.reindex e).basisOf i) j) (((b.basisOf (e.symm i)).reindex (e.subtype …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The `i`th barycentric coordinate of a point. -/
noncomputable def coord (i : ι) : P →ᵃ[k] k where
  toFun q := 1 - (b.basisOf i).sumCoords (q -ᵥ b i)
  linear := -(b.basisOf i).sumCoords
  map_vadd' q v := by
    /-
      ι : Type u_1
      ι' : Type u_2
      G : Type u_3
      G' : Type u_4
      k : Type u_5
      V : Type u_6
      P : Type u_7
      inst✝³ : AddCommGroup V
      inst✝² : AddTorsor V P
      inst✝¹ : Ring k
      inst✝ : Module k V
      b : AffineBasis ι k P
      s : Finset ι
      i✝ j : ι
      e : Equiv ι ι'
      i : ι
      q : P
      v : V
      ⊢ Eq ((fun q => HSub.hSub 1 ((b.basisOf i).sumCoords (VSub.vsub q (b i)))) (HV …
    -/
    dsimp only
    rw [vadd_vsub_assoc, LinearMap.map_add, vadd_eq_add, LinearMap.neg_apply,
      sub_add_eq_sub_sub_swap, add_comm, sub_eq_add_neg]


@[simp]
theorem linear_eq_sumCoords (i : ι) : (b.coord i).linear = -(b.basisOf i).sumCoords :=
  rfl


@[simp]
theorem coord_reindex (i : ι') : (b.reindex e).coord i = b.coord (e.symm i) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : Ring k
    inst✝ : Module k V
    b : AffineBasis ι k P
    e : Equiv ι ι'
    i : ι'
    ⊢ Eq ((b.reindex e).coord i) (b.coord (e.symm i))
  -/
  ext
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : Ring k
    inst✝ : Module k V
    b : AffineBasis ι k P
    e : Equiv ι ι'
    i : ι'
    p✝ : P
    ⊢ Eq (((b.reindex e).coord i) p✝) ((b.coord (e.symm i)) p✝)
  -/
  classical simp [AffineBasis.coord]
  /-
    🎉 no goals
  -/


@[simp]
theorem coord_apply_eq (i : ι) : b.coord i (b i) = 1 := by
  simp only [coord, Basis.coe_sumCoords, LinearEquiv.map_zero, LinearEquiv.coe_coe, sub_zero,
    AffineMap.coe_mk, Finsupp.sum_zero_index, vsub_self]


@[simp]
theorem coord_apply_ne (h : i ≠ j) : b.coord i (b j) = 0 := by
  -- Porting note:
  -- in mathlib3 we didn't need to given the `fun j => j ≠ i` argument to `Subtype.coe_mk`,
  -- but I don't think we can complain: this proof was over-golfed.
  rw [coord, AffineMap.coe_mk, ← @Subtype.coe_mk _ (fun j => j ≠ i) j h.symm, ← b.basisOf_apply,
    Basis.sumCoords_self_apply, sub_self]


theorem coord_apply [DecidableEq ι] (i j : ι) : b.coord i (b j) = if i = j then 1 else 0 := by
  /-
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : DecidableEq ι
    i j : ι
    ⊢ Eq ((b.coord i) (b j)) (ite (Eq i j) 1 0)
  -/
                                     /-
                                       🎉 no goals
                                     -/
  rcases eq_or_ne i j with h | h <;> simp [h]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem coord_apply_combination_of_mem (hi : i ∈ s) {w : ι → k} (hw : s.sum w = 1) :
    b.coord i (s.affineCombination k b w) = w i := by
  classical simp only [coord_apply, hi, Finset.affineCombination_eq_linear_combination, if_true,
      mul_boole, hw, Function.comp_apply, smul_eq_mul, s.sum_ite_eq,
      s.map_affineCombination b w hw]


@[simp]
theorem coord_apply_combination_of_not_mem (hi : i ∉ s) {w : ι → k} (hw : s.sum w = 1) :
    b.coord i (s.affineCombination k b w) = 0 := by
  classical simp only [coord_apply, hi, Finset.affineCombination_eq_linear_combination, if_false,
      mul_boole, hw, Function.comp_apply, smul_eq_mul, s.sum_ite_eq,
      s.map_affineCombination b w hw]


@[simp]
theorem sum_coord_apply_eq_one [Fintype ι] (q : P) : ∑ i, b.coord i q = 1 := by
  have hq : q ∈ affineSpan k (range b) := by
    rw [b.tot]
    exact AffineSubspace.mem_top k V q
  /-
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Fintype ι
    q : P
    hq : Membership.mem (affineSpan k (Set.range ⇑b)) q
    ⊢ Eq (Finset.univ.sum fun i => (b.coord i) q) 1
  -/
  obtain ⟨w, hw, rfl⟩ := eq_affineCombination_of_mem_affineSpan_of_fintype hq
  /-
    case intro.intro
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Fintype ι
    w : ι → k
    hw : Eq (Finset.univ.sum fun i => w i) 1
    hq : Membership.mem (affineSpan k (Set.range ⇑b)) ((Finset.affineCombination k …
    ⊢ Eq (Finset.univ.sum fun i => (b.coord i) ((Finset.affineCombination k Finset …
  -/
  convert hw
  /-
    case h.e'_2.a
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Fintype ι
    w : ι → k
    hw : Eq (Finset.univ.sum fun i => w i) 1
    hq : Membership.mem (affineSpan k (Set.range ⇑b)) ((Finset.affineCombination k …
    x✝ : ι
    a✝ : Membership.mem Finset.univ x✝
    ⊢ Eq ((b.coord x✝) ((Finset.affineCombination k Finset.univ ⇑b) w)) (w x✝)
  -/
  exact b.coord_apply_combination_of_mem (Finset.mem_univ _) hw
  /-
    🎉 no goals
  -/


@[simp]
theorem affineCombination_coord_eq_self [Fintype ι] (q : P) :
    (Finset.univ.affineCombination k b fun i => b.coord i q) = q := by
  have hq : q ∈ affineSpan k (range b) := by
    rw [b.tot]
    exact AffineSubspace.mem_top k V q
  /-
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Fintype ι
    q : P
    hq : Membership.mem (affineSpan k (Set.range ⇑b)) q
    ⊢ Eq ((Finset.affineCombination k Finset.univ ⇑b) fun i => (b.coord i) q) q
  -/
  obtain ⟨w, hw, rfl⟩ := eq_affineCombination_of_mem_affineSpan_of_fintype hq
  /-
    case intro.intro
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Fintype ι
    w : ι → k
    hw : Eq (Finset.univ.sum fun i => w i) 1
    hq : Membership.mem (affineSpan k (Set.range ⇑b)) ((Finset.affineCombination k …
    ⊢ Eq ((Finset.affineCombination k Finset.univ ⇑b) fun i => (b.coord i) ((Finse …
  -/
  congr
  /-
    case intro.intro.h.e_6.h
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Fintype ι
    w : ι → k
    hw : Eq (Finset.univ.sum fun i => w i) 1
    hq : Membership.mem (affineSpan k (Set.range ⇑b)) ((Finset.affineCombination k …
    ⊢ Eq (fun i => (b.coord i) ((Finset.affineCombination k Finset.univ ⇑b) w)) w
  -/
  ext i
  /-
    case intro.intro.h.e_6.h.h
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Fintype ι
    w : ι → k
    hw : Eq (Finset.univ.sum fun i => w i) 1
    hq : Membership.mem (affineSpan k (Set.range ⇑b)) ((Finset.affineCombination k …
    i : ι
    ⊢ Eq ((b.coord i) ((Finset.affineCombination k Finset.univ ⇑b) w)) (w i)
  -/
  exact b.coord_apply_combination_of_mem (Finset.mem_univ i) hw
  /-
    🎉 no goals
  -/


/-- A variant of `AffineBasis.affineCombination_coord_eq_self` for the special case when the
affine space is a module so we can talk about linear combinations. -/
@[simp]
theorem linear_combination_coord_eq_self [Fintype ι] (b : AffineBasis ι k V) (v : V) :
    ∑ i, b.coord i v • b i = v := by
  /-
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    inst✝³ : AddCommGroup V
    inst✝² : Ring k
    inst✝¹ : Module k V
    inst✝ : Fintype ι
    b : AffineBasis ι k V
    v : V
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul ((b.coord i) v) (b i)) v
  -/
  have hb := b.affineCombination_coord_eq_self v
  /-
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    inst✝³ : AddCommGroup V
    inst✝² : Ring k
    inst✝¹ : Module k V
    inst✝ : Fintype ι
    b : AffineBasis ι k V
    v : V
    hb : Eq ((Finset.affineCombination k Finset.univ ⇑b) fun i => (b.coord i) v) v
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul ((b.coord i) v) (b i)) v
  -/
  rwa [Finset.univ.affineCombination_eq_linear_combination _ _ (b.sum_coord_apply_eq_one v)] at hb
  /-
    🎉 no goals
  -/


theorem ext_elem [Finite ι] {q₁ q₂ : P} (h : ∀ i, b.coord i q₁ = b.coord i q₂) : q₁ = q₂ := by
  /-
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Finite ι
    q₁ q₂ : P
    h : ∀ (i : ι), Eq ((b.coord i) q₁) ((b.coord i) q₂)
    ⊢ Eq q₁ q₂
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Finite ι
    q₁ q₂ : P
    h : ∀ (i : ι), Eq ((b.coord i) q₁) ((b.coord i) q₂)
    val✝ : Fintype ι
    ⊢ Eq q₁ q₂
  -/
  rw [← b.affineCombination_coord_eq_self q₁, ← b.affineCombination_coord_eq_self q₂]
  /-
    case intro
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Finite ι
    q₁ q₂ : P
    h : ∀ (i : ι), Eq ((b.coord i) q₁) ((b.coord i) q₂)
    val✝ : Fintype ι
    ⊢ Eq ((Finset.affineCombination k Finset.univ ⇑b) fun i => (b.coord i) q₁) ((F …
  -/
  simp only [h]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_coord_of_subsingleton_eq_one [Subsingleton ι] (i : ι) : (b.coord i : P → k) = 1 := by
  /-
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Subsingleton ι
    i : ι
    ⊢ Eq (⇑(b.coord i)) 1
  -/
  ext q
  have hp : (range b).Subsingleton := by
    rw [← image_univ]
    apply Subsingleton.image
    apply subsingleton_of_subsingleton
  /-
    case h
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Subsingleton ι
    i : ι
    q : P
    hp : (Set.range ⇑b).Subsingleton
    ⊢ Eq ((b.coord i) q) (1 q)
  -/
  haveI := AffineSubspace.subsingleton_of_subsingleton_span_eq_top hp b.tot
  /-
    case h
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Subsingleton ι
    i : ι
    q : P
    hp : (Set.range ⇑b).Subsingleton
    this : Subsingleton P
    ⊢ Eq ((b.coord i) q) (1 q)
  -/
  let s : Finset ι := {i}
  /-
    case h
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Subsingleton ι
    i : ι
    q : P
    hp : (Set.range ⇑b).Subsingleton
    this : Subsingleton P
    s : Finset ι := Singleton.singleton i
    ⊢ Eq ((b.coord i) q) (1 q)
  -/
  have hi : i ∈ s := by simp [s]
  /-
    case h
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Subsingleton ι
    i : ι
    q : P
    hp : (Set.range ⇑b).Subsingleton
    this : Subsingleton P
    s : Finset ι := Singleton.singleton i
    hi : Membership.mem s i
    ⊢ Eq ((b.coord i) q) (1 q)
  -/
  have hw : s.sum (Function.const ι (1 : k)) = 1 := by simp [s]
  have hq : q = s.affineCombination k b (Function.const ι (1 : k)) := by
    simp [eq_iff_true_of_subsingleton]
  /-
    case h
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝⁴ : AddCommGroup V
    inst✝³ : AddTorsor V P
    inst✝² : Ring k
    inst✝¹ : Module k V
    b : AffineBasis ι k P
    inst✝ : Subsingleton ι
    i : ι
    q : P
    hp : (Set.range ⇑b).Subsingleton
    this : Subsingleton P
    s : Finset ι := Singleton.singleton i
    hi : Membership.mem s i
    hw : Eq (s.sum (Function.const ι 1)) 1
    hq : Eq q ((Finset.affineCombination k s ⇑b) (Function.const ι 1))
    ⊢ Eq ((b.coord i) q) (1 q)
  -/
  rw [Pi.one_apply, hq, b.coord_apply_combination_of_mem hi hw, Function.const_apply]
  /-
    🎉 no goals
  -/


theorem surjective_coord [Nontrivial ι] (i : ι) : Function.Surjective <| b.coord i := by
  classical
    intro x
    obtain ⟨j, hij⟩ := exists_ne i
    let s : Finset ι := {i, j}
    have hi : i ∈ s := by simp [s]
    let w : ι → k := fun j' => if j' = i then x else 1 - x
    have hw : s.sum w = 1 := by simp [s, w, Finset.sum_ite, Finset.filter_insert, hij,
      Finset.filter_true_of_mem, Finset.filter_false_of_mem]
    use s.affineCombination k b w
    simp [w, b.coord_apply_combination_of_mem hi hw]


/-- Barycentric coordinates as an affine map. -/
noncomputable def coords : P →ᵃ[k] ι → k where
  toFun q i := b.coord i q
  linear :=
    { toFun := fun v i => -(b.basisOf i).sumCoords v
                                /-
                                  ι : Type u_1
                                  ι' : Type u_2
                                  G : Type u_3
                                  G' : Type u_4
                                  k : Type u_5
                                  V : Type u_6
                                  P : Type u_7
                                  inst✝³ : AddCommGroup V
                                  inst✝² : AddTorsor V P
                                  inst✝¹ : Ring k
                                  inst✝ : Module k V
                                  b : AffineBasis ι k P
                                  s : Finset ι
                                  i j : ι
                                  e : Equiv ι ι'
                                  v w : V
                                  ⊢ Eq ((fun v i => Neg.neg ((b.basisOf i).sumCoords v)) (HAdd.hAdd v w)) (HAdd. …
                                -/
      map_add' := fun v w => by ext; simp only [LinearMap.map_add, Pi.add_apply, neg_add]
                                     /-
                                       🎉 no goals
                                     -/
                                 /-
                                   ι : Type u_1
                                   ι' : Type u_2
                                   G : Type u_3
                                   G' : Type u_4
                                   k : Type u_5
                                   V : Type u_6
                                   P : Type u_7
                                   inst✝³ : AddCommGroup V
                                   inst✝² : AddTorsor V P
                                   inst✝¹ : Ring k
                                   inst✝ : Module k V
                                   b : AffineBasis ι k P
                                   s : Finset ι
                                   i j : ι
                                   e : Equiv ι ι'
                                   t : k
                                   v : V
                                   ⊢ Eq ({ toFun := fun v i => Neg.neg ((b.basisOf i).sumCoords v), map_add' := ⋯ …
                                 -/
      map_smul' := fun t v => by ext; simp }
                                      /-
                                        🎉 no goals
                                      -/
                      /-
                        ι : Type u_1
                        ι' : Type u_2
                        G : Type u_3
                        G' : Type u_4
                        k : Type u_5
                        V : Type u_6
                        P : Type u_7
                        inst✝³ : AddCommGroup V
                        inst✝² : AddTorsor V P
                        inst✝¹ : Ring k
                        inst✝ : Module k V
                        b : AffineBasis ι k P
                        s : Finset ι
                        i j : ι
                        e : Equiv ι ι'
                        p : P
                        v : V
                        ⊢ Eq ((fun q i => (b.coord i) q) (HVAdd.hVAdd v p)) (HVAdd.hVAdd ({ toFun := f …
                      -/
  map_vadd' p v := by ext; simp
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem coords_apply (q : P) (i : ι) : b.coords q i = b.coord i q :=
  rfl


instance instVAdd : VAdd V (AffineBasis ι k P) where
  vadd x b :=
    { toFun := x +ᵥ ⇑b,
      ind' := b.ind'.vadd,
      tot' := by rw [Pi.vadd_def, ← vadd_set_range, ← AffineSubspace.pointwise_vadd_span, b.tot,
        AffineSubspace.pointwise_vadd_top] }


@[simp, norm_cast] lemma coe_vadd (v : V) (b : AffineBasis ι k P) : ⇑(v +ᵥ b) = v +ᵥ ⇑b := rfl


@[simp] lemma basisOf_vadd (v : V) (b : AffineBasis ι k P) : (v +ᵥ b).basisOf = b.basisOf := by
  /-
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : Ring k
    inst✝ : Module k V
    v : V
    b : AffineBasis ι k P
    ⊢ Eq (HVAdd.hVAdd v b).basisOf b.basisOf
  -/
  ext
  /-
    case h.a
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : Ring k
    inst✝ : Module k V
    v : V
    b : AffineBasis ι k P
    x✝ : ι
    i✝ : Subtype fun j => Ne j x✝
    ⊢ Eq (((HVAdd.hVAdd v b).basisOf x✝) i✝) ((b.basisOf x✝) i✝)
  -/
  simp
  /-
    🎉 no goals
  -/


instance instAddAction : AddAction V (AffineBasis ι k P) :=
  DFunLike.coe_injective.addAction _ coe_vadd


@[simp] lemma coord_vadd (v : V) (b : AffineBasis ι k P) :
    (v +ᵥ b).coord i = (b.coord i).comp (AffineEquiv.constVAdd k P v).symm := by
  /-
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : Ring k
    inst✝ : Module k V
    i : ι
    v : V
    b : AffineBasis ι k P
    ⊢ Eq ((HVAdd.hVAdd v b).coord i) ((b.coord i).comp ↑(AffineEquiv.constVAdd k P …
  -/
  ext p
  simp only [coord, ne_eq, basisOf_vadd, coe_vadd, Pi.vadd_apply, Basis.coe_sumCoords,
    AffineMap.coe_mk, AffineEquiv.constVAdd_symm, AffineMap.coe_comp, AffineEquiv.coe_toAffineMap,
    Function.comp_apply, AffineEquiv.constVAdd_apply, sub_right_inj]
  /-
    case h
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : Ring k
    inst✝ : Module k V
    i : ι
    v : V
    b : AffineBasis ι k P
    p : P
    ⊢ Eq (((b.basisOf i).repr (VSub.vsub p (HVAdd.hVAdd v (b i)))).sum fun x => id …
  -/
  congr! 1
  /-
    case h.h.e'_6
    ι : Type u_1
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : Ring k
    inst✝ : Module k V
    i : ι
    v : V
    b : AffineBasis ι k P
    p : P
    ⊢ Eq ((b.basisOf i).repr (VSub.vsub p (HVAdd.hVAdd v (b i)))) ((b.basisOf i).r …
  -/
  rw [vadd_vsub_assoc, neg_add_eq_sub, vsub_vadd_eq_vsub_sub]
  /-
    🎉 no goals
  -/


/-- In an affine space that is also a vector space, an `AffineBasis` can be scaled.

TODO: generalize to include `SMul (P ≃ᵃ[k] P) (AffineBasis ι k P)`, which acts on `P` with a `VAdd`
version of a `DistribMulAction`. -/
instance instSMul : SMul G (AffineBasis ι k V) where
  smul a b :=
    { toFun := a • ⇑b,
      ind' := b.ind'.smul,
      tot' := by
        rw [Pi.smul_def, ← smul_set_range, ← AffineSubspace.smul_span, b.tot,
          AffineSubspace.smul_top (Group.isUnit a)] }


@[simp, norm_cast] lemma coe_smul (a : G) (b : AffineBasis ι k V) : ⇑(a • b) = a • ⇑b := rfl


/-- TODO: generalize to include `SMul (P ≃ᵃ[k] P) (AffineBasis ι k P)`, which acts on `P` with a
`VAdd` version of a `DistribMulAction`. -/
instance [SMulCommClass G G' V] : SMulCommClass G G' (AffineBasis ι k V) where
  smul_comm _g _g' _b := DFunLike.ext _ _ fun _ => smul_comm _ _ _


/-- TODO: generalize to include `SMul (P ≃ᵃ[k] P) (AffineBasis ι k P)`, which acts on `P` with a
`VAdd` version of a `DistribMulAction`. -/
instance [SMul G G'] [IsScalarTower G G' V] : IsScalarTower G G' (AffineBasis ι k V) where
  smul_assoc _g _g' _b := DFunLike.ext _ _ fun _ => smul_assoc _ _ _


@[simp] lemma basisOf_smul (a : G) (b : AffineBasis ι k V) (i : ι) :
                                              /-
                                                ι : Type u_1
                                                G : Type u_3
                                                k : Type u_5
                                                V : Type u_6
                                                inst✝⁵ : AddCommGroup V
                                                inst✝⁴ : Ring k
                                                inst✝³ : Module k V
                                                inst✝² : Group G
                                                inst✝¹ : DistribMulAction G V
                                                inst✝ : SMulCommClass G k V
                                                a : G
                                                b : AffineBasis ι k V
                                                i : ι
                                                ⊢ Eq ((HSMul.hSMul a b).basisOf i) (HSMul.hSMul a (b.basisOf i))
                                              -/
    (a • b).basisOf i = a • b.basisOf i := by ext j; simp [smul_sub]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] lemma reindex_smul (a : G) (b : AffineBasis ι k V) (e : ι ≃ ι') :
    (a • b).reindex e = a • b.reindex e :=
  rfl


@[simp] lemma coord_smul (a : G) (b : AffineBasis ι k V) (i : ι) :
    (a • b).coord i = (b.coord i).comp (DistribMulAction.toLinearEquiv _ _ a).symm.toAffineMap := by
  /-
    ι : Type u_1
    G : Type u_3
    k : Type u_5
    V : Type u_6
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Ring k
    inst✝³ : Module k V
    inst✝² : Group G
    inst✝¹ : DistribMulAction G V
    inst✝ : SMulCommClass G k V
    a : G
    b : AffineBasis ι k V
    i : ι
    ⊢ Eq ((HSMul.hSMul a b).coord i) ((b.coord i).comp (↑(DistribMulAction.toLinea …
  -/
  ext v; simp [map_sub, coord]
         /-
           🎉 no goals
         -/


/-- TODO: generalize to include `SMul (P ≃ᵃ[k] P) (AffineBasis ι k P)`, which acts on `P` with a
`VAdd` version of a `DistribMulAction`. -/
instance instMulAction : MulAction G (AffineBasis ι k V) :=
  DFunLike.coe_injective.mulAction _ coe_smul


@[simp]
theorem coord_apply_centroid [CharZero k] (b : AffineBasis ι k P) {s : Finset ι} {i : ι}
    (hi : i ∈ s) : b.coord i (s.centroid k b) = (s.card : k)⁻¹ := by
  rw [Finset.centroid,
    b.coord_apply_combination_of_mem hi (s.sum_centroidWeights_eq_one_of_nonempty _ ⟨i, hi⟩),
    Finset.centroidWeights, Function.const_apply]


theorem exists_affine_subbasis {t : Set P} (ht : affineSpan k t = ⊤) :
    ∃ s ⊆ t, ∃ b : AffineBasis s k P, ⇑b = ((↑) : s → P) := by
  /-
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : DivisionRing k
    inst✝ : Module k V
    t : Set P
    ht : Eq (affineSpan k t) Top.top
    ⊢ Exists fun s => And (HasSubset.Subset s t) (Exists fun b => Eq (⇑b) Subtype. …
  -/
  obtain ⟨s, hst, h_tot, h_ind⟩ := exists_affineIndependent k V t
  /-
    case intro.intro.intro
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : DivisionRing k
    inst✝ : Module k V
    t : Set P
    ht : Eq (affineSpan k t) Top.top
    s : Set P
    hst : HasSubset.Subset s t
    h_tot : Eq (affineSpan k s) (affineSpan k t)
    h_ind : AffineIndependent k Subtype.val
    ⊢ Exists fun s => And (HasSubset.Subset s t) (Exists fun b => Eq (⇑b) Subtype. …
  -/
  refine ⟨s, hst, ⟨(↑), h_ind, ?_⟩, rfl⟩
  /-
    case intro.intro.intro
    k : Type u_5
    V : Type u_6
    P : Type u_7
    inst✝³ : AddCommGroup V
    inst✝² : AddTorsor V P
    inst✝¹ : DivisionRing k
    inst✝ : Module k V
    t : Set P
    ht : Eq (affineSpan k t) Top.top
    s : Set P
    hst : HasSubset.Subset s t
    h_tot : Eq (affineSpan k s) (affineSpan k t)
    h_ind : AffineIndependent k Subtype.val
    ⊢ Eq (affineSpan k (Set.range Subtype.val)) Top.top
  -/
  rw [Subtype.range_coe, h_tot, ht]
  /-
    🎉 no goals
  -/


theorem exists_affineBasis : ∃ (s : Set P) (b : AffineBasis (↥s) k P), ⇑b = ((↑) : s → P) :=
  let ⟨s, _, hs⟩ := exists_affine_subbasis (AffineSubspace.span_univ k V P)
  ⟨s, hs⟩


