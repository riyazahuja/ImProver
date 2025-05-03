/-- The class of additive monoids with an antidiagonal -/
class HasAntidiagonal (A : Type*) [AddMonoid A] where
  /-- The antidiagonal of an element `n` is the finset of pairs `(i, j)` such that `i + j = n`. -/
  antidiagonal : A → Finset (A × A)
  /-- A pair belongs to `antidiagonal n` iff the sum of its components is equal to `n`. -/
  mem_antidiagonal {n} {a} : a ∈ antidiagonal n ↔ a.fst + a.snd = n


attribute [simp] mem_antidiagonal


/-- All `HasAntidiagonal` instances are equal -/
instance [AddMonoid A] : Subsingleton (HasAntidiagonal A) where
  allEq := by
    /-
      A : Type u_1
      inst✝ : AddMonoid A
      ⊢ ∀ (a b : Finset.HasAntidiagonal A), Eq a b
    -/
    rintro ⟨a, ha⟩ ⟨b, hb⟩
    /-
      case mk.mk
      A : Type u_1
      inst✝ : AddMonoid A
      a : A → Finset (Prod A A)
      ha : ∀ {n : A} {a_1 : Prod A A}, Iff (Membership.mem (a n) a_1) (Eq (HAdd.hAdd …
      b : A → Finset (Prod A A)
      hb : ∀ {n : A} {a : Prod A A}, Iff (Membership.mem (b n) a) (Eq (HAdd.hAdd a.1 …
      ⊢ Eq { antidiagonal := a, mem_antidiagonal := ha } { antidiagonal := b, mem_an …
    -/
    congr with n xy
    /-
      case mk.mk.e_antidiagonal.h.h
      A : Type u_1
      inst✝ : AddMonoid A
      a : A → Finset (Prod A A)
      ha : ∀ {n : A} {a_1 : Prod A A}, Iff (Membership.mem (a n) a_1) (Eq (HAdd.hAdd …
      b : A → Finset (Prod A A)
      hb : ∀ {n : A} {a : Prod A A}, Iff (Membership.mem (b n) a) (Eq (HAdd.hAdd a.1 …
      n : A
      xy : Prod A A
      ⊢ Iff (Membership.mem (a n) xy) (Membership.mem (b n) xy)
    -/
    rw [ha, hb]
    /-
      🎉 no goals
    -/

-- The goal of this lemma is to allow to rewrite antidiagonal
-- when the decidability instances obsucate Lean

lemma hasAntidiagonal_congr (A : Type*) [AddMonoid A]
    [H1 : HasAntidiagonal A] [H2 : HasAntidiagonal A] :
                                            /-
                                              A : Type u_2
                                              inst✝ : AddMonoid A
                                              H1 H2 : Finset.HasAntidiagonal A
                                              ⊢ Eq Finset.HasAntidiagonal.antidiagonal Finset.HasAntidiagonal.antidiagonal
                                            -/
    H1.antidiagonal = H2.antidiagonal := by congr!; subsingleton
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem swap_mem_antidiagonal [AddCommMonoid A] [HasAntidiagonal A] {n : A} {xy : A × A} :
    xy.swap ∈ antidiagonal n ↔ xy ∈ antidiagonal n := by
  /-
    A : Type u_1
    inst✝¹ : AddCommMonoid A
    inst✝ : Finset.HasAntidiagonal A
    n : A
    xy : Prod A A
    ⊢ Iff (Membership.mem (Finset.HasAntidiagonal.antidiagonal n) xy.swap) (Member …
  -/
  simp [add_comm]
  /-
    🎉 no goals
  -/


@[simp] theorem map_prodComm_antidiagonal [AddCommMonoid A] [HasAntidiagonal A] {n : A} :
    (antidiagonal n).map (Equiv.prodComm A A) = antidiagonal n :=
                              /-
                                A : Type u_1
                                inst✝¹ : AddCommMonoid A
                                inst✝ : Finset.HasAntidiagonal A
                                n : A
                                x✝ : Prod A A
                                a b : A
                                ⊢ Iff (Membership.mem (Finset.map (Equiv.prodComm A A).toEmbedding (Finset.Has …
                              -/
  Finset.ext fun ⟨a, b⟩ => by simp [add_comm]
                              /-
                                🎉 no goals
                              -/


/-- See also `Finset.map_prodComm_antidiagonal`. -/
@[simp] theorem map_swap_antidiagonal [AddCommMonoid A] [HasAntidiagonal A] {n : A} :
    (antidiagonal n).map ⟨Prod.swap, Prod.swap_injective⟩ = antidiagonal n :=
  map_prodComm_antidiagonal


/-- A point in the antidiagonal is determined by its first coordinate.

See also `Finset.antidiagonal_congr'`. -/
theorem antidiagonal_congr (hp : p ∈ antidiagonal n) (hq : q ∈ antidiagonal n) :
    p = q ↔ p.1 = q.1 := by
  /-
    A : Type u_1
    inst✝¹ : AddCancelMonoid A
    inst✝ : Finset.HasAntidiagonal A
    p q : Prod A A
    n : A
    hp : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) p
    hq : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) q
    ⊢ Iff (Eq p q) (Eq p.1 q.1)
  -/
  refine ⟨congr_arg Prod.fst, fun h ↦ Prod.ext h ((add_right_inj q.fst).mp ?_)⟩
  /-
    A : Type u_1
    inst✝¹ : AddCancelMonoid A
    inst✝ : Finset.HasAntidiagonal A
    p q : Prod A A
    n : A
    hp : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) p
    hq : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) q
    h : Eq p.1 q.1
    ⊢ Eq (HAdd.hAdd q.1 p.2) (HAdd.hAdd q.1 q.2)
  -/
  rw [mem_antidiagonal] at hp hq
  /-
    A : Type u_1
    inst✝¹ : AddCancelMonoid A
    inst✝ : Finset.HasAntidiagonal A
    p q : Prod A A
    n : A
    hp : Eq (HAdd.hAdd p.1 p.2) n
    hq : Eq (HAdd.hAdd q.1 q.2) n
    h : Eq p.1 q.1
    ⊢ Eq (HAdd.hAdd q.1 p.2) (HAdd.hAdd q.1 q.2)
  -/
  rw [hq, ← h, hp]
  /-
    🎉 no goals
  -/


/-- A point in the antidiagonal is determined by its first co-ordinate (subtype version of
`Finset.antidiagonal_congr`). This lemma is used by the `ext` tactic. -/
@[ext] theorem antidiagonal_subtype_ext {p q : antidiagonal n} (h : p.val.1 = q.val.1) : p = q :=
  Subtype.ext ((antidiagonal_congr p.prop q.prop).mpr h)


/-- A point in the antidiagonal is determined by its second coordinate.

See also `Finset.antidiagonal_congr`. -/
lemma antidiagonal_congr' (hp : p ∈ antidiagonal n) (hq : q ∈ antidiagonal n) :
    p = q ↔ p.2 = q.2 := by
  /-
    A : Type u_1
    inst✝¹ : AddCancelCommMonoid A
    inst✝ : Finset.HasAntidiagonal A
    p q : Prod A A
    n : A
    hp : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) p
    hq : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) q
    ⊢ Iff (Eq p q) (Eq p.2 q.2)
  -/
  rw [← Prod.swap_inj]
  /-
    A : Type u_1
    inst✝¹ : AddCancelCommMonoid A
    inst✝ : Finset.HasAntidiagonal A
    p q : Prod A A
    n : A
    hp : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) p
    hq : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) q
    ⊢ Iff (Eq p.swap q.swap) (Eq p.2 q.2)
  -/
  exact antidiagonal_congr (swap_mem_antidiagonal.2 hp) (swap_mem_antidiagonal.2 hq)
  /-
    🎉 no goals
  -/


@[simp]
theorem antidiagonal_zero : antidiagonal (0 : A) = {(0, 0)} := by
  /-
    A : Type u_1
    inst✝¹ : CanonicallyOrderedAddCommMonoid A
    inst✝ : Finset.HasAntidiagonal A
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal 0) (Singleton.singleton { fst := 0,  …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    A : Type u_1
    inst✝¹ : CanonicallyOrderedAddCommMonoid A
    inst✝ : Finset.HasAntidiagonal A
    x y : A
    ⊢ Iff (Membership.mem (Finset.HasAntidiagonal.antidiagonal 0) { fst := x, snd  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem antidiagonal.fst_le {n : A} {kl : A × A} (hlk : kl ∈ antidiagonal n) : kl.1 ≤ n := by
  /-
    A : Type u_1
    inst✝¹ : CanonicallyOrderedAddCommMonoid A
    inst✝ : Finset.HasAntidiagonal A
    n : A
    kl : Prod A A
    hlk : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) kl
    ⊢ LE.le kl.1 n
  -/
  rw [le_iff_exists_add]
  /-
    A : Type u_1
    inst✝¹ : CanonicallyOrderedAddCommMonoid A
    inst✝ : Finset.HasAntidiagonal A
    n : A
    kl : Prod A A
    hlk : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) kl
    ⊢ Exists fun c => Eq n (HAdd.hAdd kl.1 c)
  -/
  use kl.2
  /-
    case h
    A : Type u_1
    inst✝¹ : CanonicallyOrderedAddCommMonoid A
    inst✝ : Finset.HasAntidiagonal A
    n : A
    kl : Prod A A
    hlk : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) kl
    ⊢ Eq n (HAdd.hAdd kl.1 kl.2)
  -/
  rwa [mem_antidiagonal, eq_comm] at hlk
  /-
    🎉 no goals
  -/


theorem antidiagonal.snd_le {n : A} {kl : A × A} (hlk : kl ∈ antidiagonal n) : kl.2 ≤ n := by
  /-
    A : Type u_1
    inst✝¹ : CanonicallyOrderedAddCommMonoid A
    inst✝ : Finset.HasAntidiagonal A
    n : A
    kl : Prod A A
    hlk : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) kl
    ⊢ LE.le kl.2 n
  -/
  rw [le_iff_exists_add]
  /-
    A : Type u_1
    inst✝¹ : CanonicallyOrderedAddCommMonoid A
    inst✝ : Finset.HasAntidiagonal A
    n : A
    kl : Prod A A
    hlk : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) kl
    ⊢ Exists fun c => Eq n (HAdd.hAdd kl.2 c)
  -/
  use kl.1
  /-
    case h
    A : Type u_1
    inst✝¹ : CanonicallyOrderedAddCommMonoid A
    inst✝ : Finset.HasAntidiagonal A
    n : A
    kl : Prod A A
    hlk : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) kl
    ⊢ Eq n (HAdd.hAdd kl.2 kl.1)
  -/
  rwa [mem_antidiagonal, eq_comm, add_comm] at hlk
  /-
    🎉 no goals
  -/


theorem filter_fst_eq_antidiagonal (n m : A) [DecidablePred (· = m)] [Decidable (m ≤ n)] :
    filter (fun x : A × A ↦ x.fst = m) (antidiagonal n) = if m ≤ n then {(m, n - m)} else ∅ := by
  /-
    A : Type u_1
    inst✝⁶ : CanonicallyOrderedAddCommMonoid A
    inst✝⁵ : Sub A
    inst✝⁴ : OrderedSub A
    inst✝³ : AddLeftReflectLE A
    inst✝² : Finset.HasAntidiagonal A
    n m : A
    inst✝¹ : DecidablePred fun x => Eq x m
    inst✝ : Decidable (LE.le m n)
    ⊢ Eq (Finset.filter (fun x => Eq x.1 m) (Finset.HasAntidiagonal.antidiagonal n …
  -/
  ext ⟨a, b⟩
  suffices a = m → (a + b = n ↔ m ≤ n ∧ b = n - m) by
    rw [mem_filter, mem_antidiagonal, apply_ite (fun n ↦ (a, b) ∈ n), mem_singleton,
      Prod.mk.inj_iff, ite_prop_iff_or]
    simpa [← and_assoc, @and_right_comm _ (a = _), and_congr_left_iff]
  /-
    case h.mk
    A : Type u_1
    inst✝⁶ : CanonicallyOrderedAddCommMonoid A
    inst✝⁵ : Sub A
    inst✝⁴ : OrderedSub A
    inst✝³ : AddLeftReflectLE A
    inst✝² : Finset.HasAntidiagonal A
    n m : A
    inst✝¹ : DecidablePred fun x => Eq x m
    inst✝ : Decidable (LE.le m n)
    a b : A
    ⊢ Eq a m → Iff (Eq (HAdd.hAdd a b) n) (And (LE.le m n) (Eq b (HSub.hSub n m)))
  -/
  rintro rfl
  /-
    case h.mk
    A : Type u_1
    inst✝⁶ : CanonicallyOrderedAddCommMonoid A
    inst✝⁵ : Sub A
    inst✝⁴ : OrderedSub A
    inst✝³ : AddLeftReflectLE A
    inst✝² : Finset.HasAntidiagonal A
    n a b : A
    inst✝¹ : DecidablePred fun x => Eq x a
    inst✝ : Decidable (LE.le a n)
    ⊢ Iff (Eq (HAdd.hAdd a b) n) (And (LE.le a n) (Eq b (HSub.hSub n a)))
  -/
  constructor
    /-
      case h.mk.mp
      A : Type u_1
      inst✝⁶ : CanonicallyOrderedAddCommMonoid A
      inst✝⁵ : Sub A
      inst✝⁴ : OrderedSub A
      inst✝³ : AddLeftReflectLE A
      inst✝² : Finset.HasAntidiagonal A
      n a b : A
      inst✝¹ : DecidablePred fun x => Eq x a
      inst✝ : Decidable (LE.le a n)
      ⊢ Eq (HAdd.hAdd a b) n → And (LE.le a n) (Eq b (HSub.hSub n a))
    -/
  · rintro rfl
    /-
      case h.mk.mp
      A : Type u_1
      inst✝⁶ : CanonicallyOrderedAddCommMonoid A
      inst✝⁵ : Sub A
      inst✝⁴ : OrderedSub A
      inst✝³ : AddLeftReflectLE A
      inst✝² : Finset.HasAntidiagonal A
      a b : A
      inst✝¹ : DecidablePred fun x => Eq x a
      inst✝ : Decidable (LE.le a (HAdd.hAdd a b))
      ⊢ And (LE.le a (HAdd.hAdd a b)) (Eq b (HSub.hSub (HAdd.hAdd a b) a))
    -/
    exact ⟨le_add_right le_rfl, (add_tsub_cancel_left _ _).symm⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mk.mpr
      A : Type u_1
      inst✝⁶ : CanonicallyOrderedAddCommMonoid A
      inst✝⁵ : Sub A
      inst✝⁴ : OrderedSub A
      inst✝³ : AddLeftReflectLE A
      inst✝² : Finset.HasAntidiagonal A
      n a b : A
      inst✝¹ : DecidablePred fun x => Eq x a
      inst✝ : Decidable (LE.le a n)
      ⊢ And (LE.le a n) (Eq b (HSub.hSub n a)) → Eq (HAdd.hAdd a b) n
    -/
  · rintro ⟨h, rfl⟩
    /-
      case h.mk.mpr.intro
      A : Type u_1
      inst✝⁶ : CanonicallyOrderedAddCommMonoid A
      inst✝⁵ : Sub A
      inst✝⁴ : OrderedSub A
      inst✝³ : AddLeftReflectLE A
      inst✝² : Finset.HasAntidiagonal A
      n a : A
      inst✝¹ : DecidablePred fun x => Eq x a
      inst✝ : Decidable (LE.le a n)
      h : LE.le a n
      ⊢ Eq (HAdd.hAdd a (HSub.hSub n a)) n
    -/
    exact add_tsub_cancel_of_le h
    /-
      🎉 no goals
    -/


theorem filter_snd_eq_antidiagonal (n m : A) [DecidablePred (· = m)] [Decidable (m ≤ n)] :
    filter (fun x : A × A ↦ x.snd = m) (antidiagonal n) = if m ≤ n then {(n - m, m)} else ∅ := by
  have : (fun x : A × A ↦ (x.snd = m)) ∘ Prod.swap = fun x : A × A ↦ x.fst = m := by
    ext; simp
  /-
    A : Type u_1
    inst✝⁶ : CanonicallyOrderedAddCommMonoid A
    inst✝⁵ : Sub A
    inst✝⁴ : OrderedSub A
    inst✝³ : AddLeftReflectLE A
    inst✝² : Finset.HasAntidiagonal A
    n m : A
    inst✝¹ : DecidablePred fun x => Eq x m
    inst✝ : Decidable (LE.le m n)
    this : Eq (Function.comp (fun x => Eq x.2 m) Prod.swap) fun x => Eq x.1 m
    ⊢ Eq (Finset.filter (fun x => Eq x.2 m) (Finset.HasAntidiagonal.antidiagonal n …
  -/
  rw [← map_swap_antidiagonal, filter_map]
  /-
    A : Type u_1
    inst✝⁶ : CanonicallyOrderedAddCommMonoid A
    inst✝⁵ : Sub A
    inst✝⁴ : OrderedSub A
    inst✝³ : AddLeftReflectLE A
    inst✝² : Finset.HasAntidiagonal A
    n m : A
    inst✝¹ : DecidablePred fun x => Eq x m
    inst✝ : Decidable (LE.le m n)
    this : Eq (Function.comp (fun x => Eq x.2 m) Prod.swap) fun x => Eq x.1 m
    ⊢ Eq (Finset.map { toFun := Prod.swap, inj' := ⋯ } (Finset.filter (Function.co …
  -/
  simp [this, filter_fst_eq_antidiagonal, apply_ite (Finset.map _)]
  /-
    🎉 no goals
  -/


/-- The disjoint union of antidiagonals `Σ (n : A), antidiagonal n` is equivalent to the product
    `A × A`. This is such an equivalence, obtained by mapping `(n, (k, l))` to `(k, l)`. -/
@[simps]
def sigmaAntidiagonalEquivProd [AddMonoid A] [HasAntidiagonal A] :
    (Σ n : A, antidiagonal n) ≃ A × A where
  toFun x := x.2
  invFun x := ⟨x.1 + x.2, x, mem_antidiagonal.mpr rfl⟩
  left_inv := by
    /-
      A : Type u_1
      inst✝¹ : AddMonoid A
      inst✝ : Finset.HasAntidiagonal A
      ⊢ Function.LeftInverse (fun x => ⟨HAdd.hAdd x.1 x.2, ⟨x, ⋯⟩⟩) fun x => ↑x.snd
    -/
    rintro ⟨n, ⟨k, l⟩, h⟩
    /-
      case mk.mk.mk
      A : Type u_1
      inst✝¹ : AddMonoid A
      inst✝ : Finset.HasAntidiagonal A
      n k l : A
      h : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := k, snd :=  …
      ⊢ Eq ((fun x => ⟨HAdd.hAdd x.1 x.2, ⟨x, ⋯⟩⟩) ((fun x => ↑x.snd) ⟨n, ⟨{ fst :=  …
    -/
    rw [mem_antidiagonal] at h
    /-
      case mk.mk.mk
      A : Type u_1
      inst✝¹ : AddMonoid A
      inst✝ : Finset.HasAntidiagonal A
      n k l : A
      h✝ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := k, snd := …
      h : Eq (HAdd.hAdd { fst := k, snd := l }.1 { fst := k, snd := l }.2) n
      ⊢ Eq ((fun x => ⟨HAdd.hAdd x.1 x.2, ⟨x, ⋯⟩⟩) ((fun x => ↑x.snd) ⟨n, ⟨{ fst :=  …
    -/
    exact Sigma.subtype_ext h rfl
    /-
      🎉 no goals
    -/
  right_inv _ := rfl


/-- In a canonically ordered add monoid, the antidiagonal can be construct by filtering.

Note that this is not an instance, as for some times a more efficient algorithm is available. -/
abbrev antidiagonalOfLocallyFinite : HasAntidiagonal A where
  antidiagonal n := Finset.filter (fun uv => uv.fst + uv.snd = n) (Finset.product (Iic n) (Iic n))
  mem_antidiagonal {n} {a} := by
    /-
      A✝ : Type u_1
      A : Type u_2
      inst✝² : CanonicallyOrderedAddCommMonoid A
      inst✝¹ : LocallyFiniteOrder A
      inst✝ : DecidableEq A
      n : A
      a : Prod A A
      ⊢ Iff (Membership.mem ((fun n => Finset.filter (fun uv => Eq (HAdd.hAdd uv.1 u …
    -/
    simp only [Prod.forall, mem_filter, and_iff_right_iff_imp]
    /-
      A✝ : Type u_1
      A : Type u_2
      inst✝² : CanonicallyOrderedAddCommMonoid A
      inst✝¹ : LocallyFiniteOrder A
      inst✝ : DecidableEq A
      n : A
      a : Prod A A
      ⊢ Eq (HAdd.hAdd a.1 a.2) n → Membership.mem ((Finset.Iic n).product (Finset.Ii …
    -/
    intro h; rw [← h]
    /-
      A✝ : Type u_1
      A : Type u_2
      inst✝² : CanonicallyOrderedAddCommMonoid A
      inst✝¹ : LocallyFiniteOrder A
      inst✝ : DecidableEq A
      n : A
      a : Prod A A
      h : Eq (HAdd.hAdd a.1 a.2) n
      ⊢ Membership.mem ((Finset.Iic (HAdd.hAdd a.1 a.2)).product (Finset.Iic (HAdd.h …
    -/
    erw [mem_product, mem_Iic, mem_Iic]
    /-
      A✝ : Type u_1
      A : Type u_2
      inst✝² : CanonicallyOrderedAddCommMonoid A
      inst✝¹ : LocallyFiniteOrder A
      inst✝ : DecidableEq A
      n : A
      a : Prod A A
      h : Eq (HAdd.hAdd a.1 a.2) n
      ⊢ And (LE.le a.1 (HAdd.hAdd a.1 a.2)) (LE.le a.2 (HAdd.hAdd a.1 a.2))
    -/
    exact ⟨le_self_add, le_add_self⟩
    /-
      🎉 no goals
    -/


