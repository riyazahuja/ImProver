/-- A `JordanHolderLattice` is the class for which the Jordan Hölder theorem is proved. A
Jordan Hölder lattice is a lattice equipped with a notion of maximality, `IsMaximal`, and a notion
of isomorphism of pairs `Iso`. In the example of subgroups of a group, `IsMaximal H K` means that
`H` is a maximal normal subgroup of `K`, and `Iso (H₁, K₁) (H₂, K₂)` means that the quotient
`H₁ / K₁` is isomorphic to the quotient `H₂ / K₂`. `Iso` must be symmetric and transitive and must
satisfy the second isomorphism theorem `Iso (H, H ⊔ K) (H ⊓ K, K)`.
Examples include `Subgroup G` if `G` is a group, and `Submodule R M` if `M` is an `R`-module.
-/
class JordanHolderLattice (X : Type u) [Lattice X] where
  IsMaximal : X → X → Prop
  lt_of_isMaximal : ∀ {x y}, IsMaximal x y → x < y
  sup_eq_of_isMaximal : ∀ {x y z}, IsMaximal x z → IsMaximal y z → x ≠ y → x ⊔ y = z
  isMaximal_inf_left_of_isMaximal_sup :
    ∀ {x y}, IsMaximal x (x ⊔ y) → IsMaximal y (x ⊔ y) → IsMaximal (x ⊓ y) x
  Iso : X × X → X × X → Prop
  iso_symm : ∀ {x y}, Iso x y → Iso y x
  iso_trans : ∀ {x y z}, Iso x y → Iso y z → Iso x z
  second_iso : ∀ {x y}, IsMaximal x (x ⊔ y) → Iso (x, x ⊔ y) (x ⊓ y, y)


theorem isMaximal_inf_right_of_isMaximal_sup {x y : X} (hxz : IsMaximal x (x ⊔ y))
    (hyz : IsMaximal y (x ⊔ y)) : IsMaximal (x ⊓ y) y := by
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    x y : X
    hxz : JordanHolderLattice.IsMaximal x (Max.max x y)
    hyz : JordanHolderLattice.IsMaximal y (Max.max x y)
    ⊢ JordanHolderLattice.IsMaximal (Min.min x y) y
  -/
  rw [inf_comm]
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    x y : X
    hxz : JordanHolderLattice.IsMaximal x (Max.max x y)
    hyz : JordanHolderLattice.IsMaximal y (Max.max x y)
    ⊢ JordanHolderLattice.IsMaximal (Min.min y x) y
  -/
  rw [sup_comm] at hxz hyz
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    x y : X
    hxz : JordanHolderLattice.IsMaximal x (Max.max y x)
    hyz : JordanHolderLattice.IsMaximal y (Max.max y x)
    ⊢ JordanHolderLattice.IsMaximal (Min.min y x) y
  -/
  exact isMaximal_inf_left_of_isMaximal_sup hyz hxz
  /-
    🎉 no goals
  -/


theorem isMaximal_of_eq_inf (x b : X) {a y : X} (ha : x ⊓ y = a) (hxy : x ≠ y) (hxb : IsMaximal x b)
    (hyb : IsMaximal y b) : IsMaximal a y := by
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    x b a y : X
    ha : Eq (Min.min x y) a
    hxy : Ne x y
    hxb : JordanHolderLattice.IsMaximal x b
    hyb : JordanHolderLattice.IsMaximal y b
    ⊢ JordanHolderLattice.IsMaximal a y
  -/
  have hb : x ⊔ y = b := sup_eq_of_isMaximal hxb hyb hxy
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    x b a y : X
    ha : Eq (Min.min x y) a
    hxy : Ne x y
    hxb : JordanHolderLattice.IsMaximal x b
    hyb : JordanHolderLattice.IsMaximal y b
    hb : Eq (Max.max x y) b
    ⊢ JordanHolderLattice.IsMaximal a y
  -/
  substs a b
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    x y : X
    hxy : Ne x y
    hxb : JordanHolderLattice.IsMaximal x (Max.max x y)
    hyb : JordanHolderLattice.IsMaximal y (Max.max x y)
    ⊢ JordanHolderLattice.IsMaximal (Min.min x y) y
  -/
  exact isMaximal_inf_right_of_isMaximal_sup hxb hyb
  /-
    🎉 no goals
  -/


theorem second_iso_of_eq {x y a b : X} (hm : IsMaximal x a) (ha : x ⊔ y = a) (hb : x ⊓ y = b) :
                            /-
                              X : Type u
                              inst✝¹ : Lattice X
                              inst✝ : JordanHolderLattice X
                              x y a b : X
                              hm : JordanHolderLattice.IsMaximal x a
                              ha : Eq (Max.max x y) a
                              hb : Eq (Min.min x y) b
                              ⊢ JordanHolderLattice.Iso { fst := x, snd := a } { fst := b, snd := y }
                            -/
    Iso (x, a) (b, y) := by substs a b; exact second_iso hm
                                        /-
                                          🎉 no goals
                                        -/


theorem IsMaximal.iso_refl {x y : X} (h : IsMaximal x y) : Iso (x, y) (x, y) :=
  second_iso_of_eq h (sup_eq_right.2 (le_of_lt (lt_of_isMaximal h)))
    (inf_eq_left.2 (le_of_lt (lt_of_isMaximal h)))


/-- A `CompositionSeries X` is a finite nonempty series of elements of a
`JordanHolderLattice` such that each element is maximal inside the next. The length of a
`CompositionSeries X` is one less than the number of elements in the series.
Note that there is no stipulation that a series start from the bottom of the lattice and finish at
the top. For a composition series `s`, `s.last` is the largest element of the series,
and `s.head` is the least element.
-/
abbrev CompositionSeries (X : Type u) [Lattice X] [JordanHolderLattice X] : Type u :=
  RelSeries (IsMaximal (X := X))


theorem lt_succ (s : CompositionSeries X) (i : Fin s.length) :
    s (Fin.castSucc i) < s (Fin.succ i) :=
  lt_of_isMaximal (s.step _)


protected theorem strictMono (s : CompositionSeries X) : StrictMono s :=
  Fin.strictMono_iff_lt_succ.2 s.lt_succ


protected theorem injective (s : CompositionSeries X) : Function.Injective s :=
  s.strictMono.injective


@[simp]
protected theorem inj (s : CompositionSeries X) {i j : Fin s.length.succ} : s i = s j ↔ i = j :=
  s.injective.eq_iff


theorem total {s : CompositionSeries X} {x y : X} (hx : x ∈ s) (hy : y ∈ s) : x ≤ y ∨ y ≤ x := by
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    x y : X
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ Or (LE.le x y) (LE.le y x)
  -/
  rcases Set.mem_range.1 hx with ⟨i, rfl⟩
  /-
    case intro
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    y : X
    hy : Membership.mem s y
    i : Fin (HAdd.hAdd s.length 1)
    hx : Membership.mem s (s.toFun i)
    ⊢ Or (LE.le (s.toFun i) y) (LE.le y (s.toFun i))
  -/
  rcases Set.mem_range.1 hy with ⟨j, rfl⟩
  /-
    case intro.intro
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    i : Fin (HAdd.hAdd s.length 1)
    hx : Membership.mem s (s.toFun i)
    j : Fin (HAdd.hAdd s.length 1)
    hy : Membership.mem s (s.toFun j)
    ⊢ Or (LE.le (s.toFun i) (s.toFun j)) (LE.le (s.toFun j) (s.toFun i))
  -/
  rw [s.strictMono.le_iff_le, s.strictMono.le_iff_le]
  /-
    case intro.intro
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    i : Fin (HAdd.hAdd s.length 1)
    hx : Membership.mem s (s.toFun i)
    j : Fin (HAdd.hAdd s.length 1)
    hy : Membership.mem s (s.toFun j)
    ⊢ Or (LE.le i j) (LE.le j i)
  -/
  exact le_total i j
  /-
    🎉 no goals
  -/


theorem toList_sorted (s : CompositionSeries X) : s.toList.Sorted (· < ·) :=
  List.pairwise_iff_get.2 fun i j h => by
    /-
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s : CompositionSeries X
      i j : Fin (RelSeries.toList s).length
      h : LT.lt i j
      ⊢ LT.lt ((RelSeries.toList s).get i) ((RelSeries.toList s).get j)
    -/
    dsimp only [RelSeries.toList]
    /-
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s : CompositionSeries X
      i j : Fin (RelSeries.toList s).length
      h : LT.lt i j
      ⊢ LT.lt ((List.ofFn s.toFun).get i) ((List.ofFn s.toFun).get j)
    -/
    rw [List.get_ofFn, List.get_ofFn]
    /-
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s : CompositionSeries X
      i j : Fin (RelSeries.toList s).length
      h : LT.lt i j
      ⊢ LT.lt (s.toFun (Fin.cast ⋯ i)) (s.toFun (Fin.cast ⋯ j))
    -/
    exact s.strictMono h
    /-
      🎉 no goals
    -/


theorem toList_nodup (s : CompositionSeries X) : s.toList.Nodup :=
  s.toList_sorted.nodup


/-- Two `CompositionSeries` are equal if they have the same elements. See also `ext_fun`. -/
@[ext]
theorem ext {s₁ s₂ : CompositionSeries X} (h : ∀ x, x ∈ s₁ ↔ x ∈ s₂) : s₁ = s₂ :=
  toList_injective <|
    List.eq_of_perm_of_sorted
      (by
        classical
        exact List.perm_of_nodup_nodup_toFinset_eq s₁.toList_nodup s₂.toList_nodup
          (Finset.ext <| by simpa only [List.mem_toFinset, RelSeries.mem_toList]))
      s₁.toList_sorted s₂.toList_sorted


@[simp]
theorem le_last {s : CompositionSeries X} (i : Fin (s.length + 1)) : s i ≤ s.last :=
  s.strictMono.monotone (Fin.le_last _)


theorem le_last_of_mem {s : CompositionSeries X} {x : X} (hx : x ∈ s) : x ≤ s.last :=
  let ⟨_i, hi⟩ := Set.mem_range.2 hx
  hi ▸ le_last _


@[simp]
theorem head_le {s : CompositionSeries X} (i : Fin (s.length + 1)) : s.head ≤ s i :=
  s.strictMono.monotone (Fin.zero_le _)


theorem head_le_of_mem {s : CompositionSeries X} {x : X} (hx : x ∈ s) : s.head ≤ x :=
  let ⟨_i, hi⟩ := Set.mem_range.2 hx
  hi ▸ head_le _


theorem last_eraseLast_le (s : CompositionSeries X) : s.eraseLast.last ≤ s.last := by
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    ⊢ LE.le (RelSeries.eraseLast s).last (RelSeries.last s)
  -/
  simp [eraseLast, last, s.strictMono.le_iff_le, Fin.le_iff_val_le_val, tsub_le_self]
  /-
    🎉 no goals
  -/


theorem mem_eraseLast_of_ne_of_mem {s : CompositionSeries X} {x : X}
    (hx : x ≠ s.last) (hxs : x ∈ s) : x ∈ s.eraseLast := by
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    x : X
    hx : Ne x (RelSeries.last s)
    hxs : Membership.mem s x
    ⊢ Membership.mem (RelSeries.eraseLast s) x
  -/
  rcases hxs with ⟨i, rfl⟩
  have hi : (i : ℕ) < (s.length - 1).succ := by
    conv_rhs => rw [← Nat.succ_sub (length_pos_of_nontrivial ⟨_, ⟨i, rfl⟩, _, s.last_mem, hx⟩),
      Nat.add_one_sub_one]
    exact lt_of_le_of_ne (Nat.le_of_lt_succ i.2) (by simpa [last, s.inj, Fin.ext_iff] using hx)
  /-
    case intro
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    i : Fin (HAdd.hAdd s.length 1)
    hx : Ne (s.toFun i) (RelSeries.last s)
    hi : LT.lt (↑i) (HSub.hSub s.length 1).succ
    ⊢ Membership.mem (RelSeries.eraseLast s) (s.toFun i)
  -/
  refine ⟨Fin.castSucc (n := s.length + 1) i, ?_⟩
  /-
    case intro
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    i : Fin (HAdd.hAdd s.length 1)
    hx : Ne (s.toFun i) (RelSeries.last s)
    hi : LT.lt (↑i) (HSub.hSub s.length 1).succ
    ⊢ Eq ((RelSeries.eraseLast s).toFun ↑↑i.castSucc) (s.toFun i)
  -/
  simp [Fin.ext_iff, Nat.mod_eq_of_lt hi]
  /-
    🎉 no goals
  -/


theorem mem_eraseLast {s : CompositionSeries X} {x : X} (h : 0 < s.length) :
    x ∈ s.eraseLast ↔ x ≠ s.last ∧ x ∈ s := by
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    x : X
    h : LT.lt 0 s.length
    ⊢ Iff (Membership.mem (RelSeries.eraseLast s) x) (And (Ne x (RelSeries.last s) …
  -/
  simp only [RelSeries.mem_def, eraseLast]
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    x : X
    h : LT.lt 0 s.length
    ⊢ Iff (Membership.mem (Set.range fun i => s.toFun ⟨↑i, ⋯⟩) x) (And (Ne x (RelS …
  -/
  constructor
    /-
      case mp
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s : CompositionSeries X
      x : X
      h : LT.lt 0 s.length
      ⊢ Membership.mem (Set.range fun i => s.toFun ⟨↑i, ⋯⟩) x → And (Ne x (RelSeries …
    -/
  · rintro ⟨i, rfl⟩
    have hi : (i : ℕ) < s.length := by
      conv_rhs => rw [← Nat.add_one_sub_one s.length, Nat.succ_sub h]
      exact i.2
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp [top, Fin.ext_iff, ne_of_lt hi]`.
    /-
      case mp.intro
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s : CompositionSeries X
      h : LT.lt 0 s.length
      i : Fin (HAdd.hAdd (HSub.hSub s.length 1) 1)
      hi : LT.lt (↑i) s.length
      ⊢ And (Ne ((fun i => s.toFun ⟨↑i, ⋯⟩) i) (RelSeries.last s)) (Membership.mem ( …
    -/
    simp [last, Fin.ext_iff, ne_of_lt hi, -Set.mem_range, Set.mem_range_self]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s : CompositionSeries X
      x : X
      h : LT.lt 0 s.length
      ⊢ And (Ne x (RelSeries.last s)) (Membership.mem (Set.range s.toFun) x) → Membe …
    -/
  · intro h
    /-
      case mpr
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s : CompositionSeries X
      x : X
      h✝ : LT.lt 0 s.length
      h : And (Ne x (RelSeries.last s)) (Membership.mem (Set.range s.toFun) x)
      ⊢ Membership.mem (Set.range fun i => s.toFun ⟨↑i, ⋯⟩) x
    -/
    exact mem_eraseLast_of_ne_of_mem h.1 h.2
    /-
      🎉 no goals
    -/


theorem lt_last_of_mem_eraseLast {s : CompositionSeries X} {x : X} (h : 0 < s.length)
    (hx : x ∈ s.eraseLast) : x < s.last :=
  lt_of_le_of_ne (le_last_of_mem ((mem_eraseLast h).1 hx).2) ((mem_eraseLast h).1 hx).1


theorem isMaximal_eraseLast_last {s : CompositionSeries X} (h : 0 < s.length) :
    IsMaximal s.eraseLast.last s.last := by
  have : s.length - 1 + 1 = s.length := by
    conv_rhs => rw [← Nat.add_one_sub_one s.length]; rw [Nat.succ_sub h]
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    h : LT.lt 0 s.length
    this : Eq (HAdd.hAdd (HSub.hSub s.length 1) 1) s.length
    ⊢ JordanHolderLattice.IsMaximal (RelSeries.eraseLast s).last (RelSeries.last s)
  -/
  rw [last_eraseLast, last]
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    h : LT.lt 0 s.length
    this : Eq (HAdd.hAdd (HSub.hSub s.length 1) 1) s.length
    ⊢ JordanHolderLattice.IsMaximal (s.toFun ⟨s.length.pred, ⋯⟩) (s.toFun (Fin.las …
  -/
  convert s.step ⟨s.length - 1, by omega⟩; ext; simp [this]
                                                /-
                                                  🎉 no goals
                                                -/


theorem eq_snoc_eraseLast {s : CompositionSeries X} (h : 0 < s.length) :
    s = snoc (eraseLast s) s.last (isMaximal_eraseLast_last h) := by
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    h : LT.lt 0 s.length
    ⊢ Eq s ((RelSeries.eraseLast s).snoc (RelSeries.last s) ⋯)
  -/
  ext x
  /-
    case h
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    h : LT.lt 0 s.length
    x : X
    ⊢ Iff (Membership.mem s x) (Membership.mem ((RelSeries.eraseLast s).snoc (RelS …
  -/
  simp only [mem_snoc, mem_eraseLast h, ne_eq]
  /-
    case h
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    h : LT.lt 0 s.length
    x : X
    ⊢ Iff (Membership.mem s x) (Or (And (Not (Eq x (RelSeries.last s))) (Membershi …
  -/
                              /-
                                🎉 no goals
                              -/
  by_cases h : x = s.last <;> simp [*, s.last_mem]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem snoc_eraseLast_last {s : CompositionSeries X} (h : IsMaximal s.eraseLast.last s.last) :
    s.eraseLast.snoc s.last h = s :=
  have h : 0 < s.length :=
                                                                     /-
                                                                       X : Type u
                                                                       inst✝¹ : Lattice X
                                                                       inst✝ : JordanHolderLattice X
                                                                       s : CompositionSeries X
                                                                       h : JordanHolderLattice.IsMaximal (RelSeries.eraseLast s).last (RelSeries.last …
                                                                       hs : Eq s.length 0
                                                                       ⊢ Eq (RelSeries.last s) (RelSeries.eraseLast s).last
                                                                     -/
    Nat.pos_of_ne_zero (fun hs => ne_of_gt (lt_of_isMaximal h) <| by simp [last, Fin.ext_iff, hs])
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  (eq_snoc_eraseLast h).symm


/-- Two `CompositionSeries X`, `s₁` and `s₂` are equivalent if there is a bijection
`e : Fin s₁.length ≃ Fin s₂.length` such that for any `i`,
`Iso (s₁ i) (s₁ i.succ) (s₂ (e i), s₂ (e i.succ))` -/
def Equivalent (s₁ s₂ : CompositionSeries X) : Prop :=
  ∃ f : Fin s₁.length ≃ Fin s₂.length,
    ∀ i : Fin s₁.length, Iso (s₁ (Fin.castSucc i), s₁ i.succ)
      (s₂ (Fin.castSucc (f i)), s₂ (Fin.succ (f i)))


@[refl]
theorem refl (s : CompositionSeries X) : Equivalent s s :=
  ⟨Equiv.refl _, fun _ => (s.step _).iso_refl⟩


@[symm]
theorem symm {s₁ s₂ : CompositionSeries X} (h : Equivalent s₁ s₂) : Equivalent s₂ s₁ :=
                                        /-
                                          X : Type u
                                          inst✝¹ : Lattice X
                                          inst✝ : JordanHolderLattice X
                                          s₁ s₂ : CompositionSeries X
                                          h : s₁.Equivalent s₂
                                          i : Fin s₂.length
                                          ⊢ JordanHolderLattice.Iso { fst := s₁.toFun ((Exists.choose h).symm i).castSuc …
                                        -/
  ⟨h.choose.symm, fun i => iso_symm (by simpa using h.choose_spec (h.choose.symm i))⟩
                                        /-
                                          🎉 no goals
                                        -/


@[trans]
theorem trans {s₁ s₂ s₃ : CompositionSeries X} (h₁ : Equivalent s₁ s₂) (h₂ : Equivalent s₂ s₃) :
    Equivalent s₁ s₃ :=
  ⟨h₁.choose.trans h₂.choose,
    fun i => iso_trans (h₁.choose_spec i) (h₂.choose_spec (h₁.choose i))⟩


protected theorem smash {s₁ s₂ t₁ t₂ : CompositionSeries X}
    (hs : s₁.last = s₂.head) (ht : t₁.last = t₂.head)
    (h₁ : Equivalent s₁ t₁) (h₂ : Equivalent s₂ t₂) :
    Equivalent (smash s₁ s₂ hs) (smash t₁ t₂ ht) :=
  let e : Fin (s₁.length + s₂.length) ≃ Fin (t₁.length + t₂.length) :=
    calc
      Fin (s₁.length + s₂.length) ≃ (Fin s₁.length) ⊕ (Fin s₂.length) := finSumFinEquiv.symm
      _ ≃ (Fin t₁.length) ⊕ (Fin t₂.length) := Equiv.sumCongr h₁.choose h₂.choose
      _ ≃ Fin (t₁.length + t₂.length) := finSumFinEquiv
  ⟨e, by
    /-
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s₁ s₂ t₁ t₂ : CompositionSeries X
      hs : Eq (RelSeries.last s₁) (RelSeries.head s₂)
      ht : Eq (RelSeries.last t₁) (RelSeries.head t₂)
      h₁ : s₁.Equivalent t₁
      h₂ : s₂.Equivalent t₂
      e : Equiv (Fin (HAdd.hAdd s₁.length s₂.length)) (Fin (HAdd.hAdd t₁.length t₂.l …
      ⊢ ∀ (i : Fin (RelSeries.smash s₁ s₂ hs).length), JordanHolderLattice.Iso { fst …
    -/
    intro i
    /-
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s₁ s₂ t₁ t₂ : CompositionSeries X
      hs : Eq (RelSeries.last s₁) (RelSeries.head s₂)
      ht : Eq (RelSeries.last t₁) (RelSeries.head t₂)
      h₁ : s₁.Equivalent t₁
      h₂ : s₂.Equivalent t₂
      e : Equiv (Fin (HAdd.hAdd s₁.length s₂.length)) (Fin (HAdd.hAdd t₁.length t₂.l …
      i : Fin (RelSeries.smash s₁ s₂ hs).length
      ⊢ JordanHolderLattice.Iso { fst := (RelSeries.smash s₁ s₂ hs).toFun i.castSucc …
    -/
    refine Fin.addCases ?_ ?_ i
      /-
        case refine_1
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        s₁ s₂ t₁ t₂ : CompositionSeries X
        hs : Eq (RelSeries.last s₁) (RelSeries.head s₂)
        ht : Eq (RelSeries.last t₁) (RelSeries.head t₂)
        h₁ : s₁.Equivalent t₁
        h₂ : s₂.Equivalent t₂
        e : Equiv (Fin (HAdd.hAdd s₁.length s₂.length)) (Fin (HAdd.hAdd t₁.length t₂.l …
        i : Fin (RelSeries.smash s₁ s₂ hs).length
        ⊢ ∀ (i : Fin s₁.length), JordanHolderLattice.Iso { fst := (RelSeries.smash s₁  …
      -/
    · intro i
      /-
        case refine_1
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        s₁ s₂ t₁ t₂ : CompositionSeries X
        hs : Eq (RelSeries.last s₁) (RelSeries.head s₂)
        ht : Eq (RelSeries.last t₁) (RelSeries.head t₂)
        h₁ : s₁.Equivalent t₁
        h₂ : s₂.Equivalent t₂
        e : Equiv (Fin (HAdd.hAdd s₁.length s₂.length)) (Fin (HAdd.hAdd t₁.length t₂.l …
        i✝ : Fin (RelSeries.smash s₁ s₂ hs).length
        i : Fin s₁.length
        ⊢ JordanHolderLattice.Iso { fst := (RelSeries.smash s₁ s₂ hs).toFun (Fin.castA …
      -/
      simpa [-smash_toFun, e, smash_castAdd, smash_succ_castAdd] using h₁.choose_spec i
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        s₁ s₂ t₁ t₂ : CompositionSeries X
        hs : Eq (RelSeries.last s₁) (RelSeries.head s₂)
        ht : Eq (RelSeries.last t₁) (RelSeries.head t₂)
        h₁ : s₁.Equivalent t₁
        h₂ : s₂.Equivalent t₂
        e : Equiv (Fin (HAdd.hAdd s₁.length s₂.length)) (Fin (HAdd.hAdd t₁.length t₂.l …
        i : Fin (RelSeries.smash s₁ s₂ hs).length
        ⊢ ∀ (i : Fin s₂.length), JordanHolderLattice.Iso { fst := (RelSeries.smash s₁  …
      -/
    · intro i
      /-
        case refine_2
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        s₁ s₂ t₁ t₂ : CompositionSeries X
        hs : Eq (RelSeries.last s₁) (RelSeries.head s₂)
        ht : Eq (RelSeries.last t₁) (RelSeries.head t₂)
        h₁ : s₁.Equivalent t₁
        h₂ : s₂.Equivalent t₂
        e : Equiv (Fin (HAdd.hAdd s₁.length s₂.length)) (Fin (HAdd.hAdd t₁.length t₂.l …
        i✝ : Fin (RelSeries.smash s₁ s₂ hs).length
        i : Fin s₂.length
        ⊢ JordanHolderLattice.Iso { fst := (RelSeries.smash s₁ s₂ hs).toFun (Fin.natAd …
      -/
      simpa [-smash_toFun, e, smash_natAdd, smash_succ_natAdd] using h₂.choose_spec i⟩
      /-
        🎉 no goals
      -/


protected theorem snoc {s₁ s₂ : CompositionSeries X} {x₁ x₂ : X} {hsat₁ : IsMaximal s₁.last x₁}
    {hsat₂ : IsMaximal s₂.last x₂} (hequiv : Equivalent s₁ s₂)
    (hlast : Iso (s₁.last, x₁) (s₂.last, x₂)) : Equivalent (s₁.snoc x₁ hsat₁) (s₂.snoc x₂ hsat₂) :=
  let e : Fin s₁.length.succ ≃ Fin s₂.length.succ :=
    calc
      Fin (s₁.length + 1) ≃ Option (Fin s₁.length) := finSuccEquivLast
      _ ≃ Option (Fin s₂.length) := Functor.mapEquiv Option hequiv.choose
      _ ≃ Fin (s₂.length + 1) := finSuccEquivLast.symm
  ⟨e, fun i => by
    /-
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s₁ s₂ : CompositionSeries X
      x₁ x₂ : X
      hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s₁) x₁
      hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s₂) x₂
      hequiv : s₁.Equivalent s₂
      hlast : JordanHolderLattice.Iso { fst := RelSeries.last s₁, snd := x₁ } { fst  …
      e : Equiv (Fin s₁.length.succ) (Fin s₂.length.succ) := Trans.trans (Trans.tran …
      i : Fin (RelSeries.snoc s₁ x₁ hsat₁).length
      ⊢ JordanHolderLattice.Iso { fst := (RelSeries.snoc s₁ x₁ hsat₁).toFun i.castSu …
    -/
    refine Fin.lastCases ?_ ?_ i
      /-
        case refine_1
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        s₁ s₂ : CompositionSeries X
        x₁ x₂ : X
        hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s₁) x₁
        hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s₂) x₂
        hequiv : s₁.Equivalent s₂
        hlast : JordanHolderLattice.Iso { fst := RelSeries.last s₁, snd := x₁ } { fst  …
        e : Equiv (Fin s₁.length.succ) (Fin s₂.length.succ) := Trans.trans (Trans.tran …
        i : Fin (RelSeries.snoc s₁ x₁ hsat₁).length
        ⊢ JordanHolderLattice.Iso { fst := (RelSeries.snoc s₁ x₁ hsat₁).toFun (Fin.las …
      -/
    · simpa [e, apply_last] using hlast
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        s₁ s₂ : CompositionSeries X
        x₁ x₂ : X
        hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s₁) x₁
        hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s₂) x₂
        hequiv : s₁.Equivalent s₂
        hlast : JordanHolderLattice.Iso { fst := RelSeries.last s₁, snd := x₁ } { fst  …
        e : Equiv (Fin s₁.length.succ) (Fin s₂.length.succ) := Trans.trans (Trans.tran …
        i : Fin (RelSeries.snoc s₁ x₁ hsat₁).length
        ⊢ ∀ (i : Fin (HAdd.hAdd s₁.length (RelSeries.singleton JordanHolderLattice.IsM …
      -/
    · intro i
      /-
        case refine_2
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        s₁ s₂ : CompositionSeries X
        x₁ x₂ : X
        hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s₁) x₁
        hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s₂) x₂
        hequiv : s₁.Equivalent s₂
        hlast : JordanHolderLattice.Iso { fst := RelSeries.last s₁, snd := x₁ } { fst  …
        e : Equiv (Fin s₁.length.succ) (Fin s₂.length.succ) := Trans.trans (Trans.tran …
        i✝ : Fin (RelSeries.snoc s₁ x₁ hsat₁).length
        i : Fin (HAdd.hAdd s₁.length (RelSeries.singleton JordanHolderLattice.IsMaxima …
        ⊢ JordanHolderLattice.Iso { fst := (RelSeries.snoc s₁ x₁ hsat₁).toFun i.castSu …
      -/
      simpa [e, Fin.succ_castSucc] using hequiv.choose_spec i⟩
      /-
        🎉 no goals
      -/


theorem length_eq {s₁ s₂ : CompositionSeries X} (h : Equivalent s₁ s₂) : s₁.length = s₂.length := by
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s₁ s₂ : CompositionSeries X
    h : s₁.Equivalent s₂
    ⊢ Eq s₁.length s₂.length
  -/
  simpa using Fintype.card_congr h.choose
  /-
    🎉 no goals
  -/


theorem snoc_snoc_swap {s : CompositionSeries X} {x₁ x₂ y₁ y₂ : X} {hsat₁ : IsMaximal s.last x₁}
    {hsat₂ : IsMaximal s.last x₂} {hsaty₁ : IsMaximal (snoc s x₁ hsat₁).last y₁}
    {hsaty₂ : IsMaximal (snoc s x₂ hsat₂).last y₂} (hr₁ : Iso (s.last, x₁) (x₂, y₂))
    (hr₂ : Iso (x₁, y₁) (s.last, x₂)) :
    Equivalent (snoc (snoc s x₁ hsat₁) y₁ hsaty₁) (snoc (snoc s x₂ hsat₂) y₂ hsaty₂) :=
  let e : Fin (s.length + 1 + 1) ≃ Fin (s.length + 1 + 1) :=
    Equiv.swap (Fin.last _) (Fin.castSucc (Fin.last _))
  have h1 : ∀ {i : Fin s.length},
      (Fin.castSucc (Fin.castSucc i)) ≠ (Fin.castSucc (Fin.last _)) := fun {_} =>
                 /-
                   X : Type u
                   inst✝¹ : Lattice X
                   inst✝ : JordanHolderLattice X
                   s : CompositionSeries X
                   x₁ x₂ y₁ y₂ : X
                   hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₁
                   hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₂
                   hsaty₁ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₁ hsat₁).last y₁
                   hsaty₂ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₂ hsat₂).last y₂
                   hr₁ : JordanHolderLattice.Iso { fst := RelSeries.last s, snd := x₁ } { fst :=  …
                   hr₂ : JordanHolderLattice.Iso { fst := x₁, snd := y₁ } { fst := RelSeries.last …
                   e : Equiv (Fin (HAdd.hAdd (HAdd.hAdd s.length 1) 1)) (Fin (HAdd.hAdd (HAdd.hAd …
                   x✝ : Fin s.length
                   ⊢ LT.lt x✝.castSucc.castSucc (Fin.last s.length).castSucc
                 -/
    ne_of_lt (by simp [Fin.castSucc_lt_last])
                 /-
                   🎉 no goals
                 -/
  have h2 : ∀ {i : Fin s.length},
      (Fin.castSucc (Fin.castSucc i)) ≠ Fin.last _ := fun {_} =>
                 /-
                   X : Type u
                   inst✝¹ : Lattice X
                   inst✝ : JordanHolderLattice X
                   s : CompositionSeries X
                   x₁ x₂ y₁ y₂ : X
                   hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₁
                   hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₂
                   hsaty₁ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₁ hsat₁).last y₁
                   hsaty₂ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₂ hsat₂).last y₂
                   hr₁ : JordanHolderLattice.Iso { fst := RelSeries.last s, snd := x₁ } { fst :=  …
                   hr₂ : JordanHolderLattice.Iso { fst := x₁, snd := y₁ } { fst := RelSeries.last …
                   e : Equiv (Fin (HAdd.hAdd (HAdd.hAdd s.length 1) 1)) (Fin (HAdd.hAdd (HAdd.hAd …
                   h1 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last s.length).castSucc
                   x✝ : Fin s.length
                   ⊢ LT.lt x✝.castSucc.castSucc (Fin.last (HAdd.hAdd s.length 1))
                 -/
    ne_of_lt (by simp [Fin.castSucc_lt_last])
                 /-
                   🎉 no goals
                 -/
  ⟨e, by
    /-
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s : CompositionSeries X
      x₁ x₂ y₁ y₂ : X
      hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₁
      hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₂
      hsaty₁ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₁ hsat₁).last y₁
      hsaty₂ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₂ hsat₂).last y₂
      hr₁ : JordanHolderLattice.Iso { fst := RelSeries.last s, snd := x₁ } { fst :=  …
      hr₂ : JordanHolderLattice.Iso { fst := x₁, snd := y₁ } { fst := RelSeries.last …
      e : Equiv (Fin (HAdd.hAdd (HAdd.hAdd s.length 1) 1)) (Fin (HAdd.hAdd (HAdd.hAd …
      h1 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last s.length).castSucc
      h2 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last (HAdd.hAdd s.lengt …
      ⊢ ∀ (i : Fin ((RelSeries.snoc s x₁ hsat₁).snoc y₁ hsaty₁).length), JordanHolde …
    -/
    intro i
    /-
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s : CompositionSeries X
      x₁ x₂ y₁ y₂ : X
      hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₁
      hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₂
      hsaty₁ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₁ hsat₁).last y₁
      hsaty₂ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₂ hsat₂).last y₂
      hr₁ : JordanHolderLattice.Iso { fst := RelSeries.last s, snd := x₁ } { fst :=  …
      hr₂ : JordanHolderLattice.Iso { fst := x₁, snd := y₁ } { fst := RelSeries.last …
      e : Equiv (Fin (HAdd.hAdd (HAdd.hAdd s.length 1) 1)) (Fin (HAdd.hAdd (HAdd.hAd …
      h1 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last s.length).castSucc
      h2 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last (HAdd.hAdd s.lengt …
      i : Fin ((RelSeries.snoc s x₁ hsat₁).snoc y₁ hsaty₁).length
      ⊢ JordanHolderLattice.Iso { fst := ((RelSeries.snoc s x₁ hsat₁).snoc y₁ hsaty₁ …
    -/
    dsimp only [e]
    /-
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s : CompositionSeries X
      x₁ x₂ y₁ y₂ : X
      hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₁
      hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₂
      hsaty₁ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₁ hsat₁).last y₁
      hsaty₂ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₂ hsat₂).last y₂
      hr₁ : JordanHolderLattice.Iso { fst := RelSeries.last s, snd := x₁ } { fst :=  …
      hr₂ : JordanHolderLattice.Iso { fst := x₁, snd := y₁ } { fst := RelSeries.last …
      e : Equiv (Fin (HAdd.hAdd (HAdd.hAdd s.length 1) 1)) (Fin (HAdd.hAdd (HAdd.hAd …
      h1 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last s.length).castSucc
      h2 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last (HAdd.hAdd s.lengt …
      i : Fin ((RelSeries.snoc s x₁ hsat₁).snoc y₁ hsaty₁).length
      ⊢ JordanHolderLattice.Iso { fst := ((RelSeries.snoc s x₁ hsat₁).snoc y₁ hsaty₁ …
    -/
    refine Fin.lastCases ?_ (fun i => ?_) i
    · erw [Equiv.swap_apply_left, snoc_castSucc,
      show (snoc s x₁ hsat₁).toFun (Fin.last _) = x₁ from last_snoc _ _ _, Fin.succ_last,
      show ((s.snoc x₁ hsat₁).snoc y₁ hsaty₁).toFun (Fin.last _) = y₁ from last_snoc _ _ _,
      snoc_castSucc, snoc_castSucc, Fin.succ_castSucc, snoc_castSucc, Fin.succ_last,
      show (s.snoc _ hsat₂).toFun (Fin.last _) = x₂ from last_snoc _ _ _]
      /-
        case refine_1
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        s : CompositionSeries X
        x₁ x₂ y₁ y₂ : X
        hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₁
        hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₂
        hsaty₁ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₁ hsat₁).last y₁
        hsaty₂ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₂ hsat₂).last y₂
        hr₁ : JordanHolderLattice.Iso { fst := RelSeries.last s, snd := x₁ } { fst :=  …
        hr₂ : JordanHolderLattice.Iso { fst := x₁, snd := y₁ } { fst := RelSeries.last …
        e : Equiv (Fin (HAdd.hAdd (HAdd.hAdd s.length 1) 1)) (Fin (HAdd.hAdd (HAdd.hAd …
        h1 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last s.length).castSucc
        h2 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last (HAdd.hAdd s.lengt …
        i : Fin ((RelSeries.snoc s x₁ hsat₁).snoc y₁ hsaty₁).length
        ⊢ JordanHolderLattice.Iso { fst := x₁, snd := y₁ } { fst := s.toFun (Fin.last  …
      -/
      exact hr₂
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        s : CompositionSeries X
        x₁ x₂ y₁ y₂ : X
        hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₁
        hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₂
        hsaty₁ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₁ hsat₁).last y₁
        hsaty₂ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₂ hsat₂).last y₂
        hr₁ : JordanHolderLattice.Iso { fst := RelSeries.last s, snd := x₁ } { fst :=  …
        hr₂ : JordanHolderLattice.Iso { fst := x₁, snd := y₁ } { fst := RelSeries.last …
        e : Equiv (Fin (HAdd.hAdd (HAdd.hAdd s.length 1) 1)) (Fin (HAdd.hAdd (HAdd.hAd …
        h1 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last s.length).castSucc
        h2 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last (HAdd.hAdd s.lengt …
        i✝ : Fin ((RelSeries.snoc s x₁ hsat₁).snoc y₁ hsaty₁).length
        i : Fin (HAdd.hAdd (RelSeries.snoc s x₁ hsat₁).length (RelSeries.singleton Jor …
        ⊢ JordanHolderLattice.Iso { fst := ((RelSeries.snoc s x₁ hsat₁).snoc y₁ hsaty₁ …
      -/
    · refine Fin.lastCases ?_ (fun i => ?_) i
      · erw [Equiv.swap_apply_right, snoc_castSucc, snoc_castSucc, snoc_castSucc,
          Fin.succ_castSucc, snoc_castSucc, Fin.succ_last, last_snoc', last_snoc', last_snoc']
        /-
          case refine_2.refine_1
          X : Type u
          inst✝¹ : Lattice X
          inst✝ : JordanHolderLattice X
          s : CompositionSeries X
          x₁ x₂ y₁ y₂ : X
          hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₁
          hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₂
          hsaty₁ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₁ hsat₁).last y₁
          hsaty₂ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₂ hsat₂).last y₂
          hr₁ : JordanHolderLattice.Iso { fst := RelSeries.last s, snd := x₁ } { fst :=  …
          hr₂ : JordanHolderLattice.Iso { fst := x₁, snd := y₁ } { fst := RelSeries.last …
          e : Equiv (Fin (HAdd.hAdd (HAdd.hAdd s.length 1) 1)) (Fin (HAdd.hAdd (HAdd.hAd …
          h1 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last s.length).castSucc
          h2 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last (HAdd.hAdd s.lengt …
          i✝ : Fin ((RelSeries.snoc s x₁ hsat₁).snoc y₁ hsaty₁).length
          i : Fin (HAdd.hAdd (RelSeries.snoc s x₁ hsat₁).length (RelSeries.singleton Jor …
          ⊢ JordanHolderLattice.Iso { fst := s.toFun (Fin.last (HAdd.hAdd s.length (RelS …
        -/
        exact hr₁
        /-
          🎉 no goals
        -/
      · erw [Equiv.swap_apply_of_ne_of_ne h2 h1, snoc_castSucc, snoc_castSucc,
          snoc_castSucc, snoc_castSucc, Fin.succ_castSucc, snoc_castSucc,
          Fin.succ_castSucc, snoc_castSucc, snoc_castSucc, snoc_castSucc]
        /-
          case refine_2.refine_2
          X : Type u
          inst✝¹ : Lattice X
          inst✝ : JordanHolderLattice X
          s : CompositionSeries X
          x₁ x₂ y₁ y₂ : X
          hsat₁ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₁
          hsat₂ : JordanHolderLattice.IsMaximal (RelSeries.last s) x₂
          hsaty₁ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₁ hsat₁).last y₁
          hsaty₂ : JordanHolderLattice.IsMaximal (RelSeries.snoc s x₂ hsat₂).last y₂
          hr₁ : JordanHolderLattice.Iso { fst := RelSeries.last s, snd := x₁ } { fst :=  …
          hr₂ : JordanHolderLattice.Iso { fst := x₁, snd := y₁ } { fst := RelSeries.last …
          e : Equiv (Fin (HAdd.hAdd (HAdd.hAdd s.length 1) 1)) (Fin (HAdd.hAdd (HAdd.hAd …
          h1 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last s.length).castSucc
          h2 : ∀ {i : Fin s.length}, Ne i.castSucc.castSucc (Fin.last (HAdd.hAdd s.lengt …
          i✝¹ : Fin ((RelSeries.snoc s x₁ hsat₁).snoc y₁ hsaty₁).length
          i✝ : Fin (HAdd.hAdd (RelSeries.snoc s x₁ hsat₁).length (RelSeries.singleton Jo …
          i : Fin (HAdd.hAdd s.length (RelSeries.singleton JordanHolderLattice.IsMaximal …
          ⊢ JordanHolderLattice.Iso { fst := s.toFun i.castSucc, snd := s.toFun i.succ } …
        -/
        exact (s.step i).iso_refl⟩
        /-
          🎉 no goals
        -/


theorem length_eq_zero_of_head_eq_head_of_last_eq_last_of_length_eq_zero
    {s₁ s₂ : CompositionSeries X} (hb : s₁.head = s₂.head)
    (ht : s₁.last = s₂.last) (hs₁ : s₁.length = 0) : s₂.length = 0 := by
  have : Fin.last s₂.length = (0 : Fin s₂.length.succ) :=
    s₂.injective (hb.symm.trans ((congr_arg s₁ (Fin.ext (by simp [hs₁]))).trans ht)).symm
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s₁ s₂ : CompositionSeries X
    hb : Eq (RelSeries.head s₁) (RelSeries.head s₂)
    ht : Eq (RelSeries.last s₁) (RelSeries.last s₂)
    hs₁ : Eq s₁.length 0
    this : Eq (Fin.last s₂.length) 0
    ⊢ Eq s₂.length 0
  -/
  simpa [Fin.ext_iff]
  /-
    🎉 no goals
  -/


theorem length_pos_of_head_eq_head_of_last_eq_last_of_length_pos {s₁ s₂ : CompositionSeries X}
    (hb : s₁.head = s₂.head) (ht : s₁.last = s₂.last) : 0 < s₁.length → 0 < s₂.length :=
  not_imp_not.1
    (by
      simpa only [pos_iff_ne_zero, ne_eq, Decidable.not_not] using
        length_eq_zero_of_head_eq_head_of_last_eq_last_of_length_eq_zero hb.symm ht.symm)


theorem eq_of_head_eq_head_of_last_eq_last_of_length_eq_zero {s₁ s₂ : CompositionSeries X}
    (hb : s₁.head = s₂.head) (ht : s₁.last = s₂.last) (hs₁0 : s₁.length = 0) : s₁ = s₂ := by
  have : ∀ x, x ∈ s₁ ↔ x = s₁.last := fun x =>
    ⟨fun hx =>  subsingleton_of_length_eq_zero hs₁0 hx s₁.last_mem, fun hx => hx.symm ▸ s₁.last_mem⟩
  have : ∀ x, x ∈ s₂ ↔ x = s₂.last := fun x =>
    ⟨fun hx =>
      subsingleton_of_length_eq_zero
        (length_eq_zero_of_head_eq_head_of_last_eq_last_of_length_eq_zero hb ht
          hs₁0) hx s₂.last_mem,
      fun hx => hx.symm ▸ s₂.last_mem⟩
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s₁ s₂ : CompositionSeries X
    hb : Eq (RelSeries.head s₁) (RelSeries.head s₂)
    ht : Eq (RelSeries.last s₁) (RelSeries.last s₂)
    hs₁0 : Eq s₁.length 0
    this✝ : ∀ (x : X), Iff (Membership.mem s₁ x) (Eq x (RelSeries.last s₁))
    this : ∀ (x : X), Iff (Membership.mem s₂ x) (Eq x (RelSeries.last s₂))
    ⊢ Eq s₁ s₂
  -/
  ext
  /-
    case h
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s₁ s₂ : CompositionSeries X
    hb : Eq (RelSeries.head s₁) (RelSeries.head s₂)
    ht : Eq (RelSeries.last s₁) (RelSeries.last s₂)
    hs₁0 : Eq s₁.length 0
    this✝ : ∀ (x : X), Iff (Membership.mem s₁ x) (Eq x (RelSeries.last s₁))
    this : ∀ (x : X), Iff (Membership.mem s₂ x) (Eq x (RelSeries.last s₂))
    x✝ : X
    ⊢ Iff (Membership.mem s₁ x✝) (Membership.mem s₂ x✝)
  -/
  simp [*]
  /-
    🎉 no goals
  -/


/-- Given a `CompositionSeries`, `s`, and an element `x`
such that `x` is maximal inside `s.last` there is a series, `t`,
such that `t.last = x`, `t.head = s.head`
and `snoc t s.last _` is equivalent to `s`. -/
theorem exists_last_eq_snoc_equivalent (s : CompositionSeries X) (x : X) (hm : IsMaximal x s.last)
    (hb : s.head ≤ x) :
    ∃ t : CompositionSeries X,
      t.head = s.head ∧ t.length + 1 = s.length ∧
      ∃ htx : t.last = x,
        Equivalent s (snoc t s.last (show IsMaximal t.last _ from htx.symm ▸ hm)) := by
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s : CompositionSeries X
    x : X
    hm : JordanHolderLattice.IsMaximal x (RelSeries.last s)
    hb : LE.le (RelSeries.head s) x
    ⊢ Exists fun t => And (Eq (RelSeries.head t) (RelSeries.head s)) (And (Eq (HAd …
  -/
  induction' hn : s.length with n ih generalizing s x
  · exact
      (ne_of_gt (lt_of_le_of_lt hb (lt_of_isMaximal hm))
          (subsingleton_of_length_eq_zero hn s.last_mem s.head_mem)).elim
    /-
      case succ
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      n : Nat
      ih : ∀ (s : CompositionSeries X) (x : X) (hm : JordanHolderLattice.IsMaximal x …
      s : CompositionSeries X
      x : X
      hm : JordanHolderLattice.IsMaximal x (RelSeries.last s)
      hb : LE.le (RelSeries.head s) x
      hn : Eq s.length (HAdd.hAdd n 1)
      ⊢ Exists fun t => And (Eq (RelSeries.head t) (RelSeries.head s)) (And (Eq (HAd …
    -/
  · have h0s : 0 < s.length := hn.symm ▸ Nat.succ_pos _
    /-
      case succ
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      n : Nat
      ih : ∀ (s : CompositionSeries X) (x : X) (hm : JordanHolderLattice.IsMaximal x …
      s : CompositionSeries X
      x : X
      hm : JordanHolderLattice.IsMaximal x (RelSeries.last s)
      hb : LE.le (RelSeries.head s) x
      hn : Eq s.length (HAdd.hAdd n 1)
      h0s : LT.lt 0 s.length
      ⊢ Exists fun t => And (Eq (RelSeries.head t) (RelSeries.head s)) (And (Eq (HAd …
    -/
    by_cases hetx : s.eraseLast.last = x
      /-
        case pos
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        n : Nat
        ih : ∀ (s : CompositionSeries X) (x : X) (hm : JordanHolderLattice.IsMaximal x …
        s : CompositionSeries X
        x : X
        hm : JordanHolderLattice.IsMaximal x (RelSeries.last s)
        hb : LE.le (RelSeries.head s) x
        hn : Eq s.length (HAdd.hAdd n 1)
        h0s : LT.lt 0 s.length
        hetx : Eq (RelSeries.eraseLast s).last x
        ⊢ Exists fun t => And (Eq (RelSeries.head t) (RelSeries.head s)) (And (Eq (HAd …
      -/
    · use s.eraseLast
      /-
        case h
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        n : Nat
        ih : ∀ (s : CompositionSeries X) (x : X) (hm : JordanHolderLattice.IsMaximal x …
        s : CompositionSeries X
        x : X
        hm : JordanHolderLattice.IsMaximal x (RelSeries.last s)
        hb : LE.le (RelSeries.head s) x
        hn : Eq s.length (HAdd.hAdd n 1)
        h0s : LT.lt 0 s.length
        hetx : Eq (RelSeries.eraseLast s).last x
        ⊢ And (Eq (RelSeries.eraseLast s).head (RelSeries.head s)) (And (Eq (HAdd.hAdd …
      -/
      simp [← hetx, hn]
      -- Porting note: `rfl` is required.
      /-
        case h
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        n : Nat
        ih : ∀ (s : CompositionSeries X) (x : X) (hm : JordanHolderLattice.IsMaximal x …
        s : CompositionSeries X
        x : X
        hm : JordanHolderLattice.IsMaximal x (RelSeries.last s)
        hb : LE.le (RelSeries.head s) x
        hn : Eq s.length (HAdd.hAdd n 1)
        h0s : LT.lt 0 s.length
        hetx : Eq (RelSeries.eraseLast s).last x
        ⊢ s.Equivalent s
      -/
      rfl
      /-
        🎉 no goals
      -/
    · have imxs : IsMaximal (x ⊓ s.eraseLast.last) s.eraseLast.last :=
        isMaximal_of_eq_inf x s.last rfl (Ne.symm hetx) hm (isMaximal_eraseLast_last h0s)
      /-
        case neg
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        n : Nat
        ih : ∀ (s : CompositionSeries X) (x : X) (hm : JordanHolderLattice.IsMaximal x …
        s : CompositionSeries X
        x : X
        hm : JordanHolderLattice.IsMaximal x (RelSeries.last s)
        hb : LE.le (RelSeries.head s) x
        hn : Eq s.length (HAdd.hAdd n 1)
        h0s : LT.lt 0 s.length
        hetx : Not (Eq (RelSeries.eraseLast s).last x)
        imxs : JordanHolderLattice.IsMaximal (Min.min x (RelSeries.eraseLast s).last)  …
        ⊢ Exists fun t => And (Eq (RelSeries.head t) (RelSeries.head s)) (And (Eq (HAd …
      -/
      have := ih _ _ imxs (le_inf (by simpa) (le_last_of_mem s.eraseLast.head_mem)) (by simp [hn])
      /-
        case neg
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        n : Nat
        ih : ∀ (s : CompositionSeries X) (x : X) (hm : JordanHolderLattice.IsMaximal x …
        s : CompositionSeries X
        x : X
        hm : JordanHolderLattice.IsMaximal x (RelSeries.last s)
        hb : LE.le (RelSeries.head s) x
        hn : Eq s.length (HAdd.hAdd n 1)
        h0s : LT.lt 0 s.length
        hetx : Not (Eq (RelSeries.eraseLast s).last x)
        imxs : JordanHolderLattice.IsMaximal (Min.min x (RelSeries.eraseLast s).last)  …
        this : Exists fun t => And (Eq (RelSeries.head t) (RelSeries.eraseLast s).head …
        ⊢ Exists fun t => And (Eq (RelSeries.head t) (RelSeries.head s)) (And (Eq (HAd …
      -/
      rcases this with ⟨t, htb, htl, htt, hteqv⟩
      have hmtx : IsMaximal t.last x :=
        isMaximal_of_eq_inf s.eraseLast.last s.last (by rw [inf_comm, htt]) hetx
          (isMaximal_eraseLast_last h0s) hm
      /-
        case neg.intro.intro.intro.intro
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        n : Nat
        ih : ∀ (s : CompositionSeries X) (x : X) (hm : JordanHolderLattice.IsMaximal x …
        s : CompositionSeries X
        x : X
        hm : JordanHolderLattice.IsMaximal x (RelSeries.last s)
        hb : LE.le (RelSeries.head s) x
        hn : Eq s.length (HAdd.hAdd n 1)
        h0s : LT.lt 0 s.length
        hetx : Not (Eq (RelSeries.eraseLast s).last x)
        imxs : JordanHolderLattice.IsMaximal (Min.min x (RelSeries.eraseLast s).last)  …
        t : CompositionSeries X
        htb : Eq (RelSeries.head t) (RelSeries.eraseLast s).head
        htl : Eq (HAdd.hAdd t.length 1) n
        htt : Eq (RelSeries.last t) (Min.min x (RelSeries.eraseLast s).last)
        hteqv : CompositionSeries.Equivalent (RelSeries.eraseLast s) (RelSeries.snoc t …
        hmtx : JordanHolderLattice.IsMaximal (RelSeries.last t) x
        ⊢ Exists fun t => And (Eq (RelSeries.head t) (RelSeries.head s)) (And (Eq (HAd …
      -/
      use snoc t x hmtx
      /-
        case h
        X : Type u
        inst✝¹ : Lattice X
        inst✝ : JordanHolderLattice X
        n : Nat
        ih : ∀ (s : CompositionSeries X) (x : X) (hm : JordanHolderLattice.IsMaximal x …
        s : CompositionSeries X
        x : X
        hm : JordanHolderLattice.IsMaximal x (RelSeries.last s)
        hb : LE.le (RelSeries.head s) x
        hn : Eq s.length (HAdd.hAdd n 1)
        h0s : LT.lt 0 s.length
        hetx : Not (Eq (RelSeries.eraseLast s).last x)
        imxs : JordanHolderLattice.IsMaximal (Min.min x (RelSeries.eraseLast s).last)  …
        t : CompositionSeries X
        htb : Eq (RelSeries.head t) (RelSeries.eraseLast s).head
        htl : Eq (HAdd.hAdd t.length 1) n
        htt : Eq (RelSeries.last t) (Min.min x (RelSeries.eraseLast s).last)
        hteqv : CompositionSeries.Equivalent (RelSeries.eraseLast s) (RelSeries.snoc t …
        hmtx : JordanHolderLattice.IsMaximal (RelSeries.last t) x
        ⊢ And (Eq (RelSeries.snoc t x hmtx).head (RelSeries.head s)) (And (Eq (HAdd.hA …
      -/
      refine ⟨by simp [htb], by simp [htl], by simp, ?_⟩
      have : s.Equivalent ((snoc t s.eraseLast.last <| show IsMaximal t.last _ from
        htt.symm ▸ imxs).snoc s.last
          (by simpa using isMaximal_eraseLast_last h0s)) := by
        conv_lhs => rw [eq_snoc_eraseLast h0s]
        exact Equivalent.snoc hteqv (by simpa using (isMaximal_eraseLast_last h0s).iso_refl)
      refine this.trans <| Equivalent.snoc_snoc_swap
        (iso_symm
            (second_iso_of_eq hm
              (sup_eq_of_isMaximal hm (isMaximal_eraseLast_last h0s) (Ne.symm hetx)) htt.symm))
        (second_iso_of_eq (isMaximal_eraseLast_last h0s)
            (sup_eq_of_isMaximal (isMaximal_eraseLast_last h0s) hm hetx) (by rw [inf_comm, htt]))


/-- The **Jordan-Hölder** theorem, stated for any `JordanHolderLattice`.
If two composition series start and finish at the same place, they are equivalent. -/
theorem jordan_holder (s₁ s₂ : CompositionSeries X)
    (hb : s₁.head = s₂.head) (ht : s₁.last = s₂.last) :
    Equivalent s₁ s₂ := by
  /-
    X : Type u
    inst✝¹ : Lattice X
    inst✝ : JordanHolderLattice X
    s₁ s₂ : CompositionSeries X
    hb : Eq (RelSeries.head s₁) (RelSeries.head s₂)
    ht : Eq (RelSeries.last s₁) (RelSeries.last s₂)
    ⊢ s₁.Equivalent s₂
  -/
  induction' hle : s₁.length with n ih generalizing s₁ s₂
    /-
      case zero
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      s₁ s₂ : CompositionSeries X
      hb : Eq (RelSeries.head s₁) (RelSeries.head s₂)
      ht : Eq (RelSeries.last s₁) (RelSeries.last s₂)
      hle : Eq s₁.length 0
      ⊢ s₁.Equivalent s₂
    -/
  · rw [eq_of_head_eq_head_of_last_eq_last_of_length_eq_zero hb ht hle]
    /-
      🎉 no goals
    -/
  · have h0s₂ : 0 < s₂.length :=
      length_pos_of_head_eq_head_of_last_eq_last_of_length_pos hb ht (hle.symm ▸ Nat.succ_pos _)
    rcases exists_last_eq_snoc_equivalent s₁ s₂.eraseLast.last
        (ht.symm ▸ isMaximal_eraseLast_last h0s₂)
        (hb.symm ▸ s₂.head_eraseLast ▸ head_le_of_mem (last_mem _)) with
      ⟨t, htb, htl, htt, hteq⟩
    /-
      case succ.intro.intro.intro.intro
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      n : Nat
      ih : ∀ (s₁ s₂ : CompositionSeries X), Eq (RelSeries.head s₁) (RelSeries.head s …
      s₁ s₂ : CompositionSeries X
      hb : Eq (RelSeries.head s₁) (RelSeries.head s₂)
      ht : Eq (RelSeries.last s₁) (RelSeries.last s₂)
      hle : Eq s₁.length (HAdd.hAdd n 1)
      h0s₂ : LT.lt 0 s₂.length
      t : CompositionSeries X
      htb : Eq (RelSeries.head t) (RelSeries.head s₁)
      htl : Eq (HAdd.hAdd t.length 1) s₁.length
      htt : Eq (RelSeries.last t) (RelSeries.eraseLast s₂).last
      hteq : s₁.Equivalent (RelSeries.snoc t (RelSeries.last s₁) ⋯)
      ⊢ s₁.Equivalent s₂
    -/
    have := ih t s₂.eraseLast (by simp [htb, ← hb]) htt (Nat.succ_inj'.1 (htl.trans hle))
    /-
      case succ.intro.intro.intro.intro
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      n : Nat
      ih : ∀ (s₁ s₂ : CompositionSeries X), Eq (RelSeries.head s₁) (RelSeries.head s …
      s₁ s₂ : CompositionSeries X
      hb : Eq (RelSeries.head s₁) (RelSeries.head s₂)
      ht : Eq (RelSeries.last s₁) (RelSeries.last s₂)
      hle : Eq s₁.length (HAdd.hAdd n 1)
      h0s₂ : LT.lt 0 s₂.length
      t : CompositionSeries X
      htb : Eq (RelSeries.head t) (RelSeries.head s₁)
      htl : Eq (HAdd.hAdd t.length 1) s₁.length
      htt : Eq (RelSeries.last t) (RelSeries.eraseLast s₂).last
      hteq : s₁.Equivalent (RelSeries.snoc t (RelSeries.last s₁) ⋯)
      this : t.Equivalent (RelSeries.eraseLast s₂)
      ⊢ s₁.Equivalent s₂
    -/
    refine hteq.trans ?_
    /-
      case succ.intro.intro.intro.intro
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      n : Nat
      ih : ∀ (s₁ s₂ : CompositionSeries X), Eq (RelSeries.head s₁) (RelSeries.head s …
      s₁ s₂ : CompositionSeries X
      hb : Eq (RelSeries.head s₁) (RelSeries.head s₂)
      ht : Eq (RelSeries.last s₁) (RelSeries.last s₂)
      hle : Eq s₁.length (HAdd.hAdd n 1)
      h0s₂ : LT.lt 0 s₂.length
      t : CompositionSeries X
      htb : Eq (RelSeries.head t) (RelSeries.head s₁)
      htl : Eq (HAdd.hAdd t.length 1) s₁.length
      htt : Eq (RelSeries.last t) (RelSeries.eraseLast s₂).last
      hteq : s₁.Equivalent (RelSeries.snoc t (RelSeries.last s₁) ⋯)
      this : t.Equivalent (RelSeries.eraseLast s₂)
      ⊢ CompositionSeries.Equivalent (RelSeries.snoc t (RelSeries.last s₁) ⋯) s₂
    -/
    conv_rhs => rw [eq_snoc_eraseLast h0s₂]
    /-
      case succ.intro.intro.intro.intro
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      n : Nat
      ih : ∀ (s₁ s₂ : CompositionSeries X), Eq (RelSeries.head s₁) (RelSeries.head s …
      s₁ s₂ : CompositionSeries X
      hb : Eq (RelSeries.head s₁) (RelSeries.head s₂)
      ht : Eq (RelSeries.last s₁) (RelSeries.last s₂)
      hle : Eq s₁.length (HAdd.hAdd n 1)
      h0s₂ : LT.lt 0 s₂.length
      t : CompositionSeries X
      htb : Eq (RelSeries.head t) (RelSeries.head s₁)
      htl : Eq (HAdd.hAdd t.length 1) s₁.length
      htt : Eq (RelSeries.last t) (RelSeries.eraseLast s₂).last
      hteq : s₁.Equivalent (RelSeries.snoc t (RelSeries.last s₁) ⋯)
      this : t.Equivalent (RelSeries.eraseLast s₂)
      ⊢ CompositionSeries.Equivalent (RelSeries.snoc t (RelSeries.last s₁) ⋯) ((RelS …
    -/
    simp only [ht]
    /-
      case succ.intro.intro.intro.intro
      X : Type u
      inst✝¹ : Lattice X
      inst✝ : JordanHolderLattice X
      n : Nat
      ih : ∀ (s₁ s₂ : CompositionSeries X), Eq (RelSeries.head s₁) (RelSeries.head s …
      s₁ s₂ : CompositionSeries X
      hb : Eq (RelSeries.head s₁) (RelSeries.head s₂)
      ht : Eq (RelSeries.last s₁) (RelSeries.last s₂)
      hle : Eq s₁.length (HAdd.hAdd n 1)
      h0s₂ : LT.lt 0 s₂.length
      t : CompositionSeries X
      htb : Eq (RelSeries.head t) (RelSeries.head s₁)
      htl : Eq (HAdd.hAdd t.length 1) s₁.length
      htt : Eq (RelSeries.last t) (RelSeries.eraseLast s₂).last
      hteq : s₁.Equivalent (RelSeries.snoc t (RelSeries.last s₁) ⋯)
      this : t.Equivalent (RelSeries.eraseLast s₂)
      ⊢ CompositionSeries.Equivalent (RelSeries.snoc t (RelSeries.last s₂) ⋯) ((RelS …
    -/
    exact Equivalent.snoc this (by simpa [htt] using (isMaximal_eraseLast_last h0s₂).iso_refl)
    /-
      🎉 no goals
    -/


