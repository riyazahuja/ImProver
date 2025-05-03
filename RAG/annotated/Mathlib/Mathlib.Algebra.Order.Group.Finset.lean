lemma fold_max_add [LinearOrder M] [Add M] [AddRightMono M] (s : Finset ι) (a : WithBot M)
    (f : ι → M) : s.fold max ⊥ (fun i ↦ ↑(f i) + a) = s.fold max ⊥ ((↑) ∘ f) + a := by
  classical
    induction' s using Finset.induction_on with a s _ ih <;> simp [*, max_add_add_right]


@[to_additive nsmul_inf']
lemma inf'_pow [LinearOrder M] [Monoid M] [MulLeftMono M] [MulRightMono M] (s : Finset ι)
    (f : ι → M) (n : ℕ) (hs) : s.inf' hs f ^ n = s.inf' hs fun a ↦ f a ^ n :=
  map_finset_inf' (OrderHom.mk _ <| pow_left_mono n) hs _


@[to_additive nsmul_sup']
lemma sup'_pow [LinearOrder M] [Monoid M] [MulLeftMono M] [MulRightMono M] (s : Finset ι)
    (f : ι → M) (n : ℕ) (hs) : s.sup' hs f ^ n = s.sup' hs fun a ↦ f a ^ n :=
  map_finset_sup' (OrderHom.mk _ <| pow_left_mono n) hs _


@[to_additive "Also see `Finset.sup'_add'` that works for canonically ordered monoids."]
lemma sup'_mul [MulRightMono G] (s : Finset ι) (f : ι → G) (a : G) (hs) :
    s.sup' hs f * a = s.sup' hs fun i ↦ f i * a := map_finset_sup' (OrderIso.mulRight a) hs f


set_option linter.docPrime false in
@[to_additive "Also see `Finset.add_sup''` that works for canonically ordered monoids."]
lemma mul_sup' [MulLeftMono G] (s : Finset ι) (f : ι → G) (a : G) (hs) :
    a * s.sup' hs f = s.sup' hs fun i ↦ a * f i := map_finset_sup' (OrderIso.mulLeft a) hs f


/-- Also see `Finset.sup'_add` that works for ordered groups. -/
lemma sup'_add' (s : Finset ι) (f : ι → M) (a : M) (hs : s.Nonempty) :
    s.sup' hs f + a = s.sup' hs fun i ↦ f i + a := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝³ : CanonicallyLinearOrderedAddCommMonoid M
    inst✝² : Sub M
    inst✝¹ : AddLeftReflectLE M
    inst✝ : OrderedSub M
    s : Finset ι
    f : ι → M
    a : M
    hs : s.Nonempty
    ⊢ Eq (HAdd.hAdd (s.sup' hs f) a) (s.sup' hs fun i => HAdd.hAdd (f i) a)
  -/
  apply le_antisymm
    /-
      case a
      ι : Type u_1
      M : Type u_3
      inst✝³ : CanonicallyLinearOrderedAddCommMonoid M
      inst✝² : Sub M
      inst✝¹ : AddLeftReflectLE M
      inst✝ : OrderedSub M
      s : Finset ι
      f : ι → M
      a : M
      hs : s.Nonempty
      ⊢ LE.le (HAdd.hAdd (s.sup' hs f) a) (s.sup' hs fun i => HAdd.hAdd (f i) a)
    -/
  · apply add_le_of_le_tsub_right_of_le
      /-
        case a.h
        ι : Type u_1
        M : Type u_3
        inst✝³ : CanonicallyLinearOrderedAddCommMonoid M
        inst✝² : Sub M
        inst✝¹ : AddLeftReflectLE M
        inst✝ : OrderedSub M
        s : Finset ι
        f : ι → M
        a : M
        hs : s.Nonempty
        ⊢ LE.le a (s.sup' hs fun i => HAdd.hAdd (f i) a)
      -/
    · exact Finset.le_sup'_of_le _ hs.choose_spec le_add_self
      /-
        🎉 no goals
      -/
      /-
        case a.h2
        ι : Type u_1
        M : Type u_3
        inst✝³ : CanonicallyLinearOrderedAddCommMonoid M
        inst✝² : Sub M
        inst✝¹ : AddLeftReflectLE M
        inst✝ : OrderedSub M
        s : Finset ι
        f : ι → M
        a : M
        hs : s.Nonempty
        ⊢ LE.le (s.sup' hs f) (HSub.hSub (s.sup' hs fun i => HAdd.hAdd (f i) a) a)
      -/
    · exact Finset.sup'_le _ _ fun i hi ↦ le_tsub_of_add_le_right (Finset.le_sup' (f · + a) hi)
      /-
        🎉 no goals
      -/
    /-
      case a
      ι : Type u_1
      M : Type u_3
      inst✝³ : CanonicallyLinearOrderedAddCommMonoid M
      inst✝² : Sub M
      inst✝¹ : AddLeftReflectLE M
      inst✝ : OrderedSub M
      s : Finset ι
      f : ι → M
      a : M
      hs : s.Nonempty
      ⊢ LE.le (s.sup' hs fun i => HAdd.hAdd (f i) a) (HAdd.hAdd (s.sup' hs f) a)
    -/
  · exact Finset.sup'_le _ _ fun i hi ↦ add_le_add_right (Finset.le_sup' _ hi) _
    /-
      🎉 no goals
    -/


/-- Also see `Finset.add_sup'` that works for ordered groups. -/
lemma add_sup'' (hs : s.Nonempty) (f : ι → M) (a : M) :
                                                      /-
                                                        ι : Type u_1
                                                        M : Type u_3
                                                        inst✝³ : CanonicallyLinearOrderedAddCommMonoid M
                                                        inst✝² : Sub M
                                                        inst✝¹ : AddLeftReflectLE M
                                                        inst✝ : OrderedSub M
                                                        s : Finset ι
                                                        hs : s.Nonempty
                                                        f : ι → M
                                                        a : M
                                                        ⊢ Eq (HAdd.hAdd a (s.sup' hs f)) (s.sup' hs fun i => HAdd.hAdd a (f i))
                                                      -/
    a + s.sup' hs f = s.sup' hs fun i ↦ a + f i := by simp_rw [add_comm a, Finset.sup'_add']
                                                      /-
                                                        🎉 no goals
                                                      -/


protected lemma sup_add (hs : s.Nonempty) (f : ι → M) (a : M) :
    s.sup f + a = s.sup fun i ↦ f i + a := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝³ : CanonicallyLinearOrderedAddCommMonoid M
    inst✝² : Sub M
    inst✝¹ : AddLeftReflectLE M
    inst✝ : OrderedSub M
    s : Finset ι
    hs : s.Nonempty
    f : ι → M
    a : M
    ⊢ Eq (HAdd.hAdd (s.sup f) a) (s.sup fun i => HAdd.hAdd (f i) a)
  -/
  rw [← Finset.sup'_eq_sup hs, ← Finset.sup'_eq_sup hs, sup'_add']
  /-
    🎉 no goals
  -/


protected lemma add_sup (hs : s.Nonempty) (f : ι → M) (a : M) :
    a + s.sup f = s.sup fun i ↦ a + f i := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝³ : CanonicallyLinearOrderedAddCommMonoid M
    inst✝² : Sub M
    inst✝¹ : AddLeftReflectLE M
    inst✝ : OrderedSub M
    s : Finset ι
    hs : s.Nonempty
    f : ι → M
    a : M
    ⊢ Eq (HAdd.hAdd a (s.sup f)) (s.sup fun i => HAdd.hAdd a (f i))
  -/
  rw [← Finset.sup'_eq_sup hs, ← Finset.sup'_eq_sup hs, add_sup'']
  /-
    🎉 no goals
  -/


lemma sup_add_sup (hs : s.Nonempty) (ht : t.Nonempty) (f : ι → M) (g : κ → M) :
    s.sup f + t.sup g = (s ×ˢ t).sup fun ij ↦ f ij.1 + g ij.2 := by
  /-
    ι : Type u_1
    κ : Type u_2
    M : Type u_3
    inst✝³ : CanonicallyLinearOrderedAddCommMonoid M
    inst✝² : Sub M
    inst✝¹ : AddLeftReflectLE M
    inst✝ : OrderedSub M
    s : Finset ι
    t : Finset κ
    hs : s.Nonempty
    ht : t.Nonempty
    f : ι → M
    g : κ → M
    ⊢ Eq (HAdd.hAdd (s.sup f) (t.sup g)) ((SProd.sprod s t).sup fun ij => HAdd.hAd …
  -/
  simp only [Finset.sup_add hs, Finset.add_sup ht, Finset.sup_product_left]
  /-
    🎉 no goals
  -/


