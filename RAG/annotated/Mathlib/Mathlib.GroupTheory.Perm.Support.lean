/-- Two permutations `f` and `g` are `Disjoint` if their supports are disjoint, i.e.,
every element is fixed either by `f`, or by `g`. -/
def Disjoint (f g : Perm α) :=
  ∀ x, f x = x ∨ g x = x


@[symm]
                                                          /-
                                                            α : Type u_1
                                                            f g : Equiv.Perm α
                                                            ⊢ f.Disjoint g → g.Disjoint f
                                                          -/
theorem Disjoint.symm : Disjoint f g → Disjoint g f := by simp only [Disjoint, or_comm, imp_self]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem Disjoint.symmetric : Symmetric (@Disjoint α) := fun _ _ => Disjoint.symm


instance : IsSymm (Perm α) Disjoint :=
  ⟨Disjoint.symmetric⟩


theorem disjoint_comm : Disjoint f g ↔ Disjoint g f :=
  ⟨Disjoint.symm, Disjoint.symm⟩


theorem Disjoint.commute (h : Disjoint f g) : Commute f g :=
  Equiv.ext fun x =>
    (h x).elim
      (fun hf =>
                                     /-
                                       α : Type u_1
                                       f g : Equiv.Perm α
                                       h : f.Disjoint g
                                       x : α
                                       hf : Eq (f x) x
                                       hg : Eq (f (g x)) (g x)
                                       ⊢ Eq ((HMul.hMul f g) x) ((HMul.hMul g f) x)
                                     -/
        (h (g x)).elim (fun hg => by simp [mul_apply, hf, hg]) fun hg => by
                                     /-
                                       🎉 no goals
                                     -/
          /-
            α : Type u_1
            f g : Equiv.Perm α
            h : f.Disjoint g
            x : α
            hf : Eq (f x) x
            hg : Eq (g (g x)) (g x)
            ⊢ Eq ((HMul.hMul f g) x) ((HMul.hMul g f) x)
          -/
          simp [mul_apply, hf, g.injective hg])
          /-
            🎉 no goals
          -/
      fun hg =>
                                   /-
                                     α : Type u_1
                                     f g : Equiv.Perm α
                                     h : f.Disjoint g
                                     x : α
                                     hg : Eq (g x) x
                                     hf : Eq (f (f x)) (f x)
                                     ⊢ Eq ((HMul.hMul f g) x) ((HMul.hMul g f) x)
                                   -/
      (h (f x)).elim (fun hf => by simp [mul_apply, f.injective hf, hg]) fun hf => by
                                   /-
                                     🎉 no goals
                                   -/
        /-
          α : Type u_1
          f g : Equiv.Perm α
          h : f.Disjoint g
          x : α
          hg : Eq (g x) x
          hf : Eq (g (f x)) (f x)
          ⊢ Eq ((HMul.hMul f g) x) ((HMul.hMul g f) x)
        -/
        simp [mul_apply, hf, hg]
        /-
          🎉 no goals
        -/


@[simp]
theorem disjoint_one_left (f : Perm α) : Disjoint 1 f := fun _ => Or.inl rfl


@[simp]
theorem disjoint_one_right (f : Perm α) : Disjoint f 1 := fun _ => Or.inr rfl


theorem disjoint_iff_eq_or_eq : Disjoint f g ↔ ∀ x : α, f x = x ∨ g x = x :=
  Iff.rfl


@[simp]
theorem disjoint_refl_iff : Disjoint f f ↔ f = 1 := by
  /-
    α : Type u_1
    f : Equiv.Perm α
    ⊢ Iff (f.Disjoint f) (Eq f 1)
  -/
  refine ⟨fun h => ?_, fun h => h.symm ▸ disjoint_one_left 1⟩
  /-
    α : Type u_1
    f : Equiv.Perm α
    h : f.Disjoint f
    ⊢ Eq f 1
  -/
  ext x
  /-
    case H
    α : Type u_1
    f : Equiv.Perm α
    h : f.Disjoint f
    x : α
    ⊢ Eq (f x) (1 x)
  -/
                            /-
                              🎉 no goals
                            -/
  cases' h x with hx hx <;> simp [hx]
                            /-
                              🎉 no goals
                            -/


theorem Disjoint.inv_left (h : Disjoint f g) : Disjoint f⁻¹ g := by
  /-
    α : Type u_1
    f g : Equiv.Perm α
    h : f.Disjoint g
    ⊢ (Inv.inv f).Disjoint g
  -/
  intro x
  /-
    α : Type u_1
    f g : Equiv.Perm α
    h : f.Disjoint g
    x : α
    ⊢ Or (Eq ((Inv.inv f) x) x) (Eq (g x) x)
  -/
  rw [inv_eq_iff_eq, eq_comm]
  /-
    α : Type u_1
    f g : Equiv.Perm α
    h : f.Disjoint g
    x : α
    ⊢ Or (Eq (f x) x) (Eq (g x) x)
  -/
  exact h x
  /-
    🎉 no goals
  -/


theorem Disjoint.inv_right (h : Disjoint f g) : Disjoint f g⁻¹ :=
  h.symm.inv_left.symm


@[simp]
theorem disjoint_inv_left_iff : Disjoint f⁻¹ g ↔ Disjoint f g := by
  /-
    α : Type u_1
    f g : Equiv.Perm α
    ⊢ Iff ((Inv.inv f).Disjoint g) (f.Disjoint g)
  -/
  refine ⟨fun h => ?_, Disjoint.inv_left⟩
  /-
    α : Type u_1
    f g : Equiv.Perm α
    h : (Inv.inv f).Disjoint g
    ⊢ f.Disjoint g
  -/
  convert h.inv_left
  /-
    🎉 no goals
  -/


@[simp]
theorem disjoint_inv_right_iff : Disjoint f g⁻¹ ↔ Disjoint f g := by
  /-
    α : Type u_1
    f g : Equiv.Perm α
    ⊢ Iff (f.Disjoint (Inv.inv g)) (f.Disjoint g)
  -/
  rw [disjoint_comm, disjoint_inv_left_iff, disjoint_comm]
  /-
    🎉 no goals
  -/


theorem Disjoint.mul_left (H1 : Disjoint f h) (H2 : Disjoint g h) : Disjoint (f * g) h := fun x =>
     /-
       α : Type u_1
       f g h : Equiv.Perm α
       H1 : f.Disjoint h
       H2 : g.Disjoint h
       x : α
       ⊢ Or (Eq ((HMul.hMul f g) x) x) (Eq (h x) x)
     -/
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
  by cases H1 x <;> cases H2 x <;> simp [*]
                                   /-
                                     🎉 no goals
                                   -/


theorem Disjoint.mul_right (H1 : Disjoint f g) (H2 : Disjoint f h) : Disjoint f (g * h) := by
  /-
    α : Type u_1
    f g h : Equiv.Perm α
    H1 : f.Disjoint g
    H2 : f.Disjoint h
    ⊢ f.Disjoint (HMul.hMul g h)
  -/
  rw [disjoint_comm]
  /-
    α : Type u_1
    f g h : Equiv.Perm α
    H1 : f.Disjoint g
    H2 : f.Disjoint h
    ⊢ (HMul.hMul g h).Disjoint f
  -/
  exact H1.symm.mul_left H2.symm
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: make it `@[simp]`

theorem disjoint_conj (h : Perm α) : Disjoint (h * f * h⁻¹) (h * g * h⁻¹) ↔ Disjoint f g :=
                                  /-
                                    α : Type u_1
                                    f g h : Equiv.Perm α
                                    x✝ : α
                                    ⊢ Iff (Or (Eq ((HMul.hMul (HMul.hMul h f) (Inv.inv h)) x✝) x✝) (Eq ((HMul.hMul …
                                  -/
  (h⁻¹).forall_congr fun {_} ↦ by simp only [mul_apply, eq_inv_iff_eq]
                                  /-
                                    🎉 no goals
                                  -/


theorem Disjoint.conj (H : Disjoint f g) (h : Perm α) : Disjoint (h * f * h⁻¹) (h * g * h⁻¹) :=
  (disjoint_conj h).2 H


theorem disjoint_prod_right (l : List (Perm α)) (h : ∀ g ∈ l, Disjoint f g) :
    Disjoint f l.prod := by
  /-
    α : Type u_1
    f : Equiv.Perm α
    l : List (Equiv.Perm α)
    h : ∀ (g : Equiv.Perm α), Membership.mem l g → f.Disjoint g
    ⊢ f.Disjoint l.prod
  -/
  induction' l with g l ih
    /-
      case nil
      α : Type u_1
      f : Equiv.Perm α
      h : ∀ (g : Equiv.Perm α), Membership.mem List.nil g → f.Disjoint g
      ⊢ f.Disjoint List.nil.prod
    -/
  · exact disjoint_one_right _
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      f g : Equiv.Perm α
      l : List (Equiv.Perm α)
      ih : (∀ (g : Equiv.Perm α), Membership.mem l g → f.Disjoint g) → f.Disjoint l. …
      h : ∀ (g_1 : Equiv.Perm α), Membership.mem (List.cons g l) g_1 → f.Disjoint g_1
      ⊢ f.Disjoint (List.cons g l).prod
    -/
  · rw [List.prod_cons]
    /-
      case cons
      α : Type u_1
      f g : Equiv.Perm α
      l : List (Equiv.Perm α)
      ih : (∀ (g : Equiv.Perm α), Membership.mem l g → f.Disjoint g) → f.Disjoint l. …
      h : ∀ (g_1 : Equiv.Perm α), Membership.mem (List.cons g l) g_1 → f.Disjoint g_1
      ⊢ f.Disjoint (HMul.hMul g l.prod)
    -/
    exact (h _ (List.mem_cons_self _ _)).mul_right (ih fun g hg => h g (List.mem_cons_of_mem _ hg))
    /-
      🎉 no goals
    -/


theorem disjoint_noncommProd_right {ι : Type*} {k : ι → Perm α} {s : Finset ι}
    (hs : Set.Pairwise s fun i j ↦ Commute (k i) (k j))
    (hg : ∀ i ∈ s, g.Disjoint (k i)) :
    Disjoint g (s.noncommProd k (hs)) :=
  noncommProd_induction s k hs g.Disjoint (fun _ _ ↦ Disjoint.mul_right) (disjoint_one_right g) hg


open scoped List in
theorem disjoint_prod_perm {l₁ l₂ : List (Perm α)} (hl : l₁.Pairwise Disjoint) (hp : l₁ ~ l₂) :
    l₁.prod = l₂.prod :=
  hp.prod_eq' <| hl.imp Disjoint.commute


theorem nodup_of_pairwise_disjoint {l : List (Perm α)} (h1 : (1 : Perm α) ∉ l)
    (h2 : l.Pairwise Disjoint) : l.Nodup := by
  /-
    α : Type u_1
    l : List (Equiv.Perm α)
    h1 : Not (Membership.mem l 1)
    h2 : List.Pairwise Equiv.Perm.Disjoint l
    ⊢ l.Nodup
  -/
  refine List.Pairwise.imp_of_mem ?_ h2
  /-
    α : Type u_1
    l : List (Equiv.Perm α)
    h1 : Not (Membership.mem l 1)
    h2 : List.Pairwise Equiv.Perm.Disjoint l
    ⊢ ∀ {a b : Equiv.Perm α}, Membership.mem l a → Membership.mem l b → a.Disjoint …
  -/
  intro τ σ h_mem _ h_disjoint _
  /-
    α : Type u_1
    l : List (Equiv.Perm α)
    h1 : Not (Membership.mem l 1)
    h2 : List.Pairwise Equiv.Perm.Disjoint l
    τ σ : Equiv.Perm α
    h_mem : Membership.mem l τ
    a✝¹ : Membership.mem l σ
    h_disjoint : τ.Disjoint σ
    a✝ : Eq τ σ
    ⊢ False
  -/
  subst τ
  suffices (σ : Perm α) = 1 by
    rw [this] at h_mem
    exact h1 h_mem
  /-
    α : Type u_1
    l : List (Equiv.Perm α)
    h1 : Not (Membership.mem l 1)
    h2 : List.Pairwise Equiv.Perm.Disjoint l
    σ : Equiv.Perm α
    a✝ h_mem : Membership.mem l σ
    h_disjoint : σ.Disjoint σ
    ⊢ Eq σ 1
  -/
  exact ext fun a => or_self_iff.mp (h_disjoint a)
  /-
    🎉 no goals
  -/


theorem pow_apply_eq_self_of_apply_eq_self {x : α} (hfx : f x = x) : ∀ n : ℕ, (f ^ n) x = x
  | 0 => rfl
                /-
                  α : Type u_1
                  f : Equiv.Perm α
                  x : α
                  hfx : Eq (f x) x
                  n : Nat
                  ⊢ Eq ((HPow.hPow f (HAdd.hAdd n 1)) x) x
                -/
  | n + 1 => by rw [pow_succ, mul_apply, hfx, pow_apply_eq_self_of_apply_eq_self hfx n]
                /-
                  🎉 no goals
                -/


theorem zpow_apply_eq_self_of_apply_eq_self {x : α} (hfx : f x = x) : ∀ n : ℤ, (f ^ n) x = x
  | (n : ℕ) => pow_apply_eq_self_of_apply_eq_self hfx n
                        /-
                          α : Type u_1
                          f : Equiv.Perm α
                          x : α
                          hfx : Eq (f x) x
                          n : Nat
                          ⊢ Eq ((HPow.hPow f (Int.negSucc n)) x) x
                        -/
  | Int.negSucc n => by rw [zpow_negSucc, inv_eq_iff_eq, pow_apply_eq_self_of_apply_eq_self hfx]
                        /-
                          🎉 no goals
                        -/


theorem pow_apply_eq_of_apply_apply_eq_self {x : α} (hffx : f (f x) = x) :
    ∀ n : ℕ, (f ^ n) x = x ∨ (f ^ n) x = f x
  | 0 => Or.inl rfl
  | n + 1 =>
    (pow_apply_eq_of_apply_apply_eq_self hffx n).elim
                           /-
                             α : Type u_1
                             f : Equiv.Perm α
                             x : α
                             hffx : Eq (f (f x)) x
                             n : Nat
                             h : Eq ((HPow.hPow f n) x) x
                             ⊢ Eq ((HPow.hPow f (HAdd.hAdd n 1)) x) (f x)
                           -/
      (fun h => Or.inr (by rw [pow_succ', mul_apply, h]))
                           /-
                             🎉 no goals
                           -/
                          /-
                            α : Type u_1
                            f : Equiv.Perm α
                            x : α
                            hffx : Eq (f (f x)) x
                            n : Nat
                            h : Eq ((HPow.hPow f n) x) (f x)
                            ⊢ Eq ((HPow.hPow f (HAdd.hAdd n 1)) x) x
                          -/
      fun h => Or.inl (by rw [pow_succ', mul_apply, h, hffx])
                          /-
                            🎉 no goals
                          -/


theorem zpow_apply_eq_of_apply_apply_eq_self {x : α} (hffx : f (f x) = x) :
    ∀ i : ℤ, (f ^ i) x = x ∨ (f ^ i) x = f x
  | (n : ℕ) => pow_apply_eq_of_apply_apply_eq_self hffx n
  | Int.negSucc n => by
    rw [zpow_negSucc, inv_eq_iff_eq, ← f.injective.eq_iff, ← mul_apply, ← pow_succ', eq_comm,
      inv_eq_iff_eq, ← mul_apply, ← pow_succ, @eq_comm _ x, or_comm]
    /-
      α : Type u_1
      f : Equiv.Perm α
      x : α
      hffx : Eq (f (f x)) x
      n : Nat
      ⊢ Or (Eq ((HPow.hPow f (HAdd.hAdd (HAdd.hAdd n 1) 1)) x) x) (Eq ((HPow.hPow f  …
    -/
    exact pow_apply_eq_of_apply_apply_eq_self hffx _
    /-
      🎉 no goals
    -/


theorem Disjoint.mul_apply_eq_iff {σ τ : Perm α} (hστ : Disjoint σ τ) {a : α} :
    (σ * τ) a = a ↔ σ a = a ∧ τ a = a := by
  /-
    α : Type u_1
    σ τ : Equiv.Perm α
    hστ : σ.Disjoint τ
    a : α
    ⊢ Iff (Eq ((HMul.hMul σ τ) a) a) (And (Eq (σ a) a) (Eq (τ a) a))
  -/
  refine ⟨fun h => ?_, fun h => by rw [mul_apply, h.2, h.1]⟩
  /-
    α : Type u_1
    σ τ : Equiv.Perm α
    hστ : σ.Disjoint τ
    a : α
    h : Eq ((HMul.hMul σ τ) a) a
    ⊢ And (Eq (σ a) a) (Eq (τ a) a)
  -/
  cases' hστ a with hσ hτ
    /-
      case inl
      α : Type u_1
      σ τ : Equiv.Perm α
      hστ : σ.Disjoint τ
      a : α
      h : Eq ((HMul.hMul σ τ) a) a
      hσ : Eq (σ a) a
      ⊢ And (Eq (σ a) a) (Eq (τ a) a)
    -/
  · exact ⟨hσ, σ.injective (h.trans hσ.symm)⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      σ τ : Equiv.Perm α
      hστ : σ.Disjoint τ
      a : α
      h : Eq ((HMul.hMul σ τ) a) a
      hτ : Eq (τ a) a
      ⊢ And (Eq (σ a) a) (Eq (τ a) a)
    -/
  · exact ⟨(congr_arg σ hτ).symm.trans h, hτ⟩
    /-
      🎉 no goals
    -/


theorem Disjoint.mul_eq_one_iff {σ τ : Perm α} (hστ : Disjoint σ τ) :
    σ * τ = 1 ↔ σ = 1 ∧ τ = 1 := by
  /-
    α : Type u_1
    σ τ : Equiv.Perm α
    hστ : σ.Disjoint τ
    ⊢ Iff (Eq (HMul.hMul σ τ) 1) (And (Eq σ 1) (Eq τ 1))
  -/
  simp_rw [Perm.ext_iff, one_apply, hστ.mul_apply_eq_iff, forall_and]
  /-
    🎉 no goals
  -/


theorem Disjoint.zpow_disjoint_zpow {σ τ : Perm α} (hστ : Disjoint σ τ) (m n : ℤ) :
    Disjoint (σ ^ m) (τ ^ n) := fun x =>
  Or.imp (fun h => zpow_apply_eq_self_of_apply_eq_self h m)
    (fun h => zpow_apply_eq_self_of_apply_eq_self h n) (hστ x)


theorem Disjoint.pow_disjoint_pow {σ τ : Perm α} (hστ : Disjoint σ τ) (m n : ℕ) :
    Disjoint (σ ^ m) (τ ^ n) :=
  hστ.zpow_disjoint_zpow m n


/-- `f.IsSwap` indicates that the permutation `f` is a transposition of two elements. -/
def IsSwap (f : Perm α) : Prop :=
  ∃ x y, x ≠ y ∧ f = swap x y


@[simp]
theorem ofSubtype_swap_eq {p : α → Prop} [DecidablePred p] (x y : Subtype p) :
    ofSubtype (Equiv.swap x y) = Equiv.swap ↑x ↑y :=
  Equiv.ext fun z => by
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      p : α → Prop
      inst✝ : DecidablePred p
      x y : Subtype p
      z : α
      ⊢ Eq ((Equiv.Perm.ofSubtype (Equiv.swap x y)) z) ((Equiv.swap ↑x ↑y) z)
    -/
    by_cases hz : p z
      /-
        case pos
        α : Type u_1
        inst✝¹ : DecidableEq α
        p : α → Prop
        inst✝ : DecidablePred p
        x y : Subtype p
        z : α
        hz : p z
        ⊢ Eq ((Equiv.Perm.ofSubtype (Equiv.swap x y)) z) ((Equiv.swap ↑x ↑y) z)
      -/
    · rw [swap_apply_def, ofSubtype_apply_of_mem _ hz]
      /-
        case pos
        α : Type u_1
        inst✝¹ : DecidableEq α
        p : α → Prop
        inst✝ : DecidablePred p
        x y : Subtype p
        z : α
        hz : p z
        ⊢ Eq (↑((Equiv.swap x y) ⟨z, hz⟩)) (ite (Eq z ↑x) (↑y) (ite (Eq z ↑y) (↑x) z))
      -/
      split_ifs with hzx hzy
        /-
          case pos
          α : Type u_1
          inst✝¹ : DecidableEq α
          p : α → Prop
          inst✝ : DecidablePred p
          x y : Subtype p
          z : α
          hz : p z
          hzx : Eq z ↑x
          ⊢ Eq ↑((Equiv.swap x y) ⟨z, hz⟩) ↑y
        -/
      · simp_rw [hzx, Subtype.coe_eta, swap_apply_left]
        /-
          🎉 no goals
        -/
        /-
          case pos
          α : Type u_1
          inst✝¹ : DecidableEq α
          p : α → Prop
          inst✝ : DecidablePred p
          x y : Subtype p
          z : α
          hz : p z
          hzx : Not (Eq z ↑x)
          hzy : Eq z ↑y
          ⊢ Eq ↑((Equiv.swap x y) ⟨z, hz⟩) ↑x
        -/
      · simp_rw [hzy, Subtype.coe_eta, swap_apply_right]
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          inst✝¹ : DecidableEq α
          p : α → Prop
          inst✝ : DecidablePred p
          x y : Subtype p
          z : α
          hz : p z
          hzx : Not (Eq z ↑x)
          hzy : Not (Eq z ↑y)
          ⊢ Eq (↑((Equiv.swap x y) ⟨z, hz⟩)) z
        -/
      · rw [swap_apply_of_ne_of_ne] <;>
        /-
          case neg.a
          α : Type u_1
          inst✝¹ : DecidableEq α
          p : α → Prop
          inst✝ : DecidablePred p
          x y : Subtype p
          z : α
          hz : p z
          hzx : Not (Eq z ↑x)
          hzy : Not (Eq z ↑y)
          ⊢ Ne ⟨z, hz⟩ x
        -/
        /-
          🎉 no goals
        -/
        simp [Subtype.ext_iff, *]
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        inst✝¹ : DecidableEq α
        p : α → Prop
        inst✝ : DecidablePred p
        x y : Subtype p
        z : α
        hz : Not (p z)
        ⊢ Eq ((Equiv.Perm.ofSubtype (Equiv.swap x y)) z) ((Equiv.swap ↑x ↑y) z)
      -/
    · rw [ofSubtype_apply_of_not_mem _ hz, swap_apply_of_ne_of_ne]
        /-
          case neg.a
          α : Type u_1
          inst✝¹ : DecidableEq α
          p : α → Prop
          inst✝ : DecidablePred p
          x y : Subtype p
          z : α
          hz : Not (p z)
          ⊢ Ne z ↑x
        -/
      · intro h
        /-
          case neg.a
          α : Type u_1
          inst✝¹ : DecidableEq α
          p : α → Prop
          inst✝ : DecidablePred p
          x y : Subtype p
          z : α
          hz : Not (p z)
          h : Eq z ↑x
          ⊢ False
        -/
        apply hz
        /-
          case neg.a
          α : Type u_1
          inst✝¹ : DecidableEq α
          p : α → Prop
          inst✝ : DecidablePred p
          x y : Subtype p
          z : α
          hz : Not (p z)
          h : Eq z ↑x
          ⊢ p z
        -/
        rw [h]
        /-
          case neg.a
          α : Type u_1
          inst✝¹ : DecidableEq α
          p : α → Prop
          inst✝ : DecidablePred p
          x y : Subtype p
          z : α
          hz : Not (p z)
          h : Eq z ↑x
          ⊢ p ↑x
        -/
        exact Subtype.prop x
        /-
          🎉 no goals
        -/
      /-
        case neg.a
        α : Type u_1
        inst✝¹ : DecidableEq α
        p : α → Prop
        inst✝ : DecidablePred p
        x y : Subtype p
        z : α
        hz : Not (p z)
        ⊢ Ne z ↑y
      -/
      intro h
      /-
        case neg.a
        α : Type u_1
        inst✝¹ : DecidableEq α
        p : α → Prop
        inst✝ : DecidablePred p
        x y : Subtype p
        z : α
        hz : Not (p z)
        h : Eq z ↑y
        ⊢ False
      -/
      apply hz
      /-
        case neg.a
        α : Type u_1
        inst✝¹ : DecidableEq α
        p : α → Prop
        inst✝ : DecidablePred p
        x y : Subtype p
        z : α
        hz : Not (p z)
        h : Eq z ↑y
        ⊢ p z
      -/
      rw [h]
      /-
        case neg.a
        α : Type u_1
        inst✝¹ : DecidableEq α
        p : α → Prop
        inst✝ : DecidablePred p
        x y : Subtype p
        z : α
        hz : Not (p z)
        h : Eq z ↑y
        ⊢ p ↑y
      -/
      exact Subtype.prop y
      /-
        🎉 no goals
      -/


theorem IsSwap.of_subtype_isSwap {p : α → Prop} [DecidablePred p] {f : Perm (Subtype p)}
    (h : f.IsSwap) : (ofSubtype f).IsSwap :=
  let ⟨⟨x, hx⟩, ⟨y, hy⟩, hxy⟩ := h
  ⟨x, y, by
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      p : α → Prop
      inst✝ : DecidablePred p
      f : Equiv.Perm (Subtype p)
      h : f.IsSwap
      x : α
      hx : p x
      y : α
      hy : p y
      hxy : And (Ne ⟨x, hx⟩ ⟨y, hy⟩) (Eq f (Equiv.swap ⟨x, hx⟩ ⟨y, hy⟩))
      ⊢ Ne x y
    -/
    simp only [Ne, Subtype.ext_iff] at hxy
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      p : α → Prop
      inst✝ : DecidablePred p
      f : Equiv.Perm (Subtype p)
      h : f.IsSwap
      x : α
      hx : p x
      y : α
      hy : p y
      hxy : And (Not (Eq x y)) (Eq f (Equiv.swap ⟨x, hx⟩ ⟨y, hy⟩))
      ⊢ Ne x y
    -/
    exact hxy.1, by
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      p : α → Prop
      inst✝ : DecidablePred p
      f : Equiv.Perm (Subtype p)
      h : f.IsSwap
      x : α
      hx : p x
      y : α
      hy : p y
      hxy : And (Ne ⟨x, hx⟩ ⟨y, hy⟩) (Eq f (Equiv.swap ⟨x, hx⟩ ⟨y, hy⟩))
      ⊢ Eq (Equiv.Perm.ofSubtype f) (Equiv.swap x y)
    -/
    rw [hxy.2, ofSubtype_swap_eq]⟩
    /-
      🎉 no goals
    -/


theorem ne_and_ne_of_swap_mul_apply_ne_self {f : Perm α} {x y : α} (hy : (swap x (f x) * f) y ≠ y) :
    f y ≠ y ∧ y ≠ x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    x y : α
    hy : Ne ((HMul.hMul (Equiv.swap x (f x)) f) y) y
    ⊢ And (Ne (f y) y) (Ne y x)
  -/
  simp only [swap_apply_def, mul_apply, f.injective.eq_iff] at *
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    x y : α
    hy : Ne (ite (Eq (f y) x) (f x) (ite (Eq y x) x (f y))) y
    ⊢ And (Ne (f y) y) (Ne y x)
  -/
  by_cases h : f y = x
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x y : α
      hy : Ne (ite (Eq (f y) x) (f x) (ite (Eq y x) x (f y))) y
      h : Eq (f y) x
      ⊢ And (Ne (f y) y) (Ne y x)
    -/
                              /-
                                🎉 no goals
                              -/
  · constructor <;> intro <;> simp_all only [if_true, eq_self_iff_true, not_true, Ne]
                              /-
                                🎉 no goals
                              -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      x y : α
      hy : Ne (ite (Eq (f y) x) (f x) (ite (Eq y x) x (f y))) y
      h : Not (Eq (f y) x)
      ⊢ And (Ne (f y) y) (Ne y x)
    -/
                               /-
                                 🎉 no goals
                               -/
  · split_ifs at hy with h <;> try { simp [*] at * }
                               /-
                                 🎉 no goals
                               -/


theorem set_support_inv_eq : { x | p⁻¹ x ≠ x } = { x | p x ≠ x } := by
  /-
    α : Type u_1
    p : Equiv.Perm α
    ⊢ Eq (setOf fun x => Ne ((Inv.inv p) x) x) (setOf fun x => Ne (p x) x)
  -/
  ext x
  /-
    case h
    α : Type u_1
    p : Equiv.Perm α
    x : α
    ⊢ Iff (Membership.mem (setOf fun x => Ne ((Inv.inv p) x) x) x) (Membership.mem …
  -/
  simp only [Set.mem_setOf_eq, Ne]
  /-
    case h
    α : Type u_1
    p : Equiv.Perm α
    x : α
    ⊢ Iff (Not (Eq ((Inv.inv p) x) x)) (Not (Eq (p x) x))
  -/
  rw [inv_def, symm_apply_eq, eq_comm]
  /-
    🎉 no goals
  -/


theorem set_support_apply_mem {p : Perm α} {a : α} :
                                                      /-
                                                        α : Type u_1
                                                        p : Equiv.Perm α
                                                        a : α
                                                        ⊢ Iff (Membership.mem (setOf fun x => Ne (p x) x) (p a)) (Membership.mem (setO …
                                                      -/
    p a ∈ { x | p x ≠ x } ↔ a ∈ { x | p x ≠ x } := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem set_support_zpow_subset (n : ℤ) : { x | (p ^ n) x ≠ x } ⊆ { x | p x ≠ x } := by
  /-
    α : Type u_1
    p : Equiv.Perm α
    n : Int
    ⊢ HasSubset.Subset (setOf fun x => Ne ((HPow.hPow p n) x) x) (setOf fun x => N …
  -/
  intro x
  /-
    α : Type u_1
    p : Equiv.Perm α
    n : Int
    x : α
    ⊢ Membership.mem (setOf fun x => Ne ((HPow.hPow p n) x) x) x → Membership.mem  …
  -/
  simp only [Set.mem_setOf_eq, Ne]
  /-
    α : Type u_1
    p : Equiv.Perm α
    n : Int
    x : α
    ⊢ Not (Eq ((HPow.hPow p n) x) x) → Not (Eq (p x) x)
  -/
  intro hx H
  /-
    α : Type u_1
    p : Equiv.Perm α
    n : Int
    x : α
    hx : Not (Eq ((HPow.hPow p n) x) x)
    H : Eq (p x) x
    ⊢ False
  -/
  simp [zpow_apply_eq_self_of_apply_eq_self H] at hx
  /-
    🎉 no goals
  -/


theorem set_support_mul_subset : { x | (p * q) x ≠ x } ⊆ { x | p x ≠ x } ∪ { x | q x ≠ x } := by
  /-
    α : Type u_1
    p q : Equiv.Perm α
    ⊢ HasSubset.Subset (setOf fun x => Ne ((HMul.hMul p q) x) x) (Union.union (set …
  -/
  intro x
  /-
    α : Type u_1
    p q : Equiv.Perm α
    x : α
    ⊢ Membership.mem (setOf fun x => Ne ((HMul.hMul p q) x) x) x → Membership.mem  …
  -/
  simp only [Perm.coe_mul, Function.comp_apply, Ne, Set.mem_union, Set.mem_setOf_eq]
  /-
    α : Type u_1
    p q : Equiv.Perm α
    x : α
    ⊢ Not (Eq (p (q x)) x) → Or (Not (Eq (p x) x)) (Not (Eq (q x) x))
  -/
                            /-
                              🎉 no goals
                            -/
  by_cases hq : q x = x <;> simp [hq]
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem apply_pow_apply_eq_iff (f : Perm α) (n : ℕ) {x : α} :
    f ((f ^ n) x) = (f ^ n) x ↔ f x = x := by
  /-
    α : Type u_1
    f : Equiv.Perm α
    n : Nat
    x : α
    ⊢ Iff (Eq (f ((HPow.hPow f n) x)) ((HPow.hPow f n) x)) (Eq (f x) x)
  -/
  rw [← mul_apply, Commute.self_pow f, mul_apply, apply_eq_iff_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem apply_zpow_apply_eq_iff (f : Perm α) (n : ℤ) {x : α} :
    f ((f ^ n) x) = (f ^ n) x ↔ f x = x := by
  /-
    α : Type u_1
    f : Equiv.Perm α
    n : Int
    x : α
    ⊢ Iff (Eq (f ((HPow.hPow f n) x)) ((HPow.hPow f n) x)) (Eq (f x) x)
  -/
  rw [← mul_apply, Commute.self_zpow f, mul_apply, apply_eq_iff_eq]
  /-
    🎉 no goals
  -/


/-- The `Finset` of nonfixed points of a permutation. -/
def support (f : Perm α) : Finset α := {x | f x ≠ x}


@[simp]
theorem mem_support {x : α} : x ∈ f.support ↔ f x ≠ x := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x : α
    ⊢ Iff (Membership.mem f.support x) (Ne (f x) x)
  -/
  rw [support, mem_filter, and_iff_right (mem_univ x)]
  /-
    🎉 no goals
  -/


                                                                /-
                                                                  α : Type u_1
                                                                  inst✝¹ : DecidableEq α
                                                                  inst✝ : Fintype α
                                                                  f : Equiv.Perm α
                                                                  x : α
                                                                  ⊢ Iff (Not (Membership.mem f.support x)) (Eq (f x) x)
                                                                -/
theorem not_mem_support {x : α} : x ∉ f.support ↔ f x = x := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem coe_support_eq_set_support (f : Perm α) : (f.support : Set α) = { x | f x ≠ x } := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    ⊢ Eq (↑f.support) (setOf fun x => Ne (f x) x)
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x✝ : α
    ⊢ Iff (Membership.mem (↑f.support) x✝) (Membership.mem (setOf fun x => Ne (f x …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem support_eq_empty_iff {σ : Perm α} : σ.support = ∅ ↔ σ = 1 := by
  simp_rw [Finset.ext_iff, mem_support, Finset.not_mem_empty, iff_false, not_not,
    Equiv.Perm.ext_iff, one_apply]


@[simp]
                                                     /-
                                                       α : Type u_1
                                                       inst✝¹ : DecidableEq α
                                                       inst✝ : Fintype α
                                                       ⊢ Eq (Equiv.Perm.support 1) EmptyCollection.emptyCollection
                                                     -/
theorem support_one : (1 : Perm α).support = ∅ := by rw [support_eq_empty_iff]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem support_refl : support (Equiv.refl α) = ∅ :=
  support_one


theorem support_congr (h : f.support ⊆ g.support) (h' : ∀ x ∈ g.support, f x = g x) : f = g := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : HasSubset.Subset f.support g.support
    h' : ∀ (x : α), Membership.mem g.support x → Eq (f x) (g x)
    ⊢ Eq f g
  -/
  ext x
  /-
    case H
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : HasSubset.Subset f.support g.support
    h' : ∀ (x : α), Membership.mem g.support x → Eq (f x) (g x)
    x : α
    ⊢ Eq (f x) (g x)
  -/
  by_cases hx : x ∈ g.support
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : HasSubset.Subset f.support g.support
      h' : ∀ (x : α), Membership.mem g.support x → Eq (f x) (g x)
      x : α
      hx : Membership.mem g.support x
      ⊢ Eq (f x) (g x)
    -/
  · exact h' x hx
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : HasSubset.Subset f.support g.support
      h' : ∀ (x : α), Membership.mem g.support x → Eq (f x) (g x)
      x : α
      hx : Not (Membership.mem g.support x)
      ⊢ Eq (f x) (g x)
    -/
  · rw [not_mem_support.mp hx, ← not_mem_support]
    /-
      case neg
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : HasSubset.Subset f.support g.support
      h' : ∀ (x : α), Membership.mem g.support x → Eq (f x) (g x)
      x : α
      hx : Not (Membership.mem g.support x)
      ⊢ Not (Membership.mem f.support x)
    -/
    exact fun H => hx (h H)
    /-
      🎉 no goals
    -/


/-- If g and c commute, then g stabilizes the support of c -/
theorem mem_support_iff_of_commute {g c : Perm α} (hgc : Commute g c) (x : α) :
    x ∈ c.support ↔ g x ∈ c.support := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hgc : Commute g c
    x : α
    ⊢ Iff (Membership.mem c.support x) (Membership.mem c.support (g x))
  -/
  simp only [mem_support, not_iff_not, ← mul_apply]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    hgc : Commute g c
    x : α
    ⊢ Iff (Eq (c x) x) (Eq ((HMul.hMul c g) x) (g x))
  -/
  rw [← hgc, mul_apply, Equiv.apply_eq_iff_eq]
  /-
    🎉 no goals
  -/


theorem support_mul_le (f g : Perm α) : (f * g).support ≤ f.support ⊔ g.support := fun x => by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    x : α
    ⊢ Membership.mem (HMul.hMul f g).support x → Membership.mem (Max.max f.support …
  -/
  simp only [sup_eq_union]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    x : α
    ⊢ Membership.mem (HMul.hMul f g).support x → Membership.mem (Union.union f.sup …
  -/
  rw [mem_union, mem_support, mem_support, mem_support, mul_apply, ← not_and_or, not_imp_not]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    x : α
    ⊢ And (Eq (f x) x) (Eq (g x) x) → Eq (f (g x)) x
  -/
  rintro ⟨hf, hg⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    x : α
    hf : Eq (f x) x
    hg : Eq (g x) x
    ⊢ Eq (f (g x)) x
  -/
  rw [hg, hf]
  /-
    🎉 no goals
  -/


theorem exists_mem_support_of_mem_support_prod {l : List (Perm α)} {x : α}
    (hx : x ∈ l.prod.support) : ∃ f : Perm α, f ∈ l ∧ x ∈ f.support := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l : List (Equiv.Perm α)
    x : α
    hx : Membership.mem l.prod.support x
    ⊢ Exists fun f => And (Membership.mem l f) (Membership.mem f.support x)
  -/
  contrapose! hx
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l : List (Equiv.Perm α)
    x : α
    hx : ∀ (f : Equiv.Perm α), Membership.mem l f → Not (Membership.mem f.support x)
    ⊢ Not (Membership.mem l.prod.support x)
  -/
  simp_rw [mem_support, not_not] at hx ⊢
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l : List (Equiv.Perm α)
    x : α
    hx : ∀ (f : Equiv.Perm α), Membership.mem l f → Eq (f x) x
    ⊢ Eq (l.prod x) x
  -/
  induction' l with f l ih
    /-
      case nil
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x : α
      hx : ∀ (f : Equiv.Perm α), Membership.mem List.nil f → Eq (f x) x
      ⊢ Eq (List.nil.prod x) x
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x : α
      f : Equiv.Perm α
      l : List (Equiv.Perm α)
      ih : (∀ (f : Equiv.Perm α), Membership.mem l f → Eq (f x) x) → Eq (l.prod x) x
      hx : ∀ (f_1 : Equiv.Perm α), Membership.mem (List.cons f l) f_1 → Eq (f_1 x) x
      ⊢ Eq ((List.cons f l).prod x) x
    -/
  · rw [List.prod_cons, mul_apply, ih, hx]
      /-
        case cons._
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        x : α
        f : Equiv.Perm α
        l : List (Equiv.Perm α)
        ih : (∀ (f : Equiv.Perm α), Membership.mem l f → Eq (f x) x) → Eq (l.prod x) x
        hx : ∀ (f_1 : Equiv.Perm α), Membership.mem (List.cons f l) f_1 → Eq (f_1 x) x
        ⊢ Membership.mem (List.cons f l) f
      -/
    · simp only [List.find?, List.mem_cons, true_or]
      /-
        🎉 no goals
      -/
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x : α
      f : Equiv.Perm α
      l : List (Equiv.Perm α)
      ih : (∀ (f : Equiv.Perm α), Membership.mem l f → Eq (f x) x) → Eq (l.prod x) x
      hx : ∀ (f_1 : Equiv.Perm α), Membership.mem (List.cons f l) f_1 → Eq (f_1 x) x
      ⊢ ∀ (f : Equiv.Perm α), Membership.mem l f → Eq (f x) x
    -/
    intros f' hf'
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x : α
      f : Equiv.Perm α
      l : List (Equiv.Perm α)
      ih : (∀ (f : Equiv.Perm α), Membership.mem l f → Eq (f x) x) → Eq (l.prod x) x
      hx : ∀ (f_1 : Equiv.Perm α), Membership.mem (List.cons f l) f_1 → Eq (f_1 x) x
      f' : Equiv.Perm α
      hf' : Membership.mem l f'
      ⊢ Eq (f' x) x
    -/
    refine hx f' ?_
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x : α
      f : Equiv.Perm α
      l : List (Equiv.Perm α)
      ih : (∀ (f : Equiv.Perm α), Membership.mem l f → Eq (f x) x) → Eq (l.prod x) x
      hx : ∀ (f_1 : Equiv.Perm α), Membership.mem (List.cons f l) f_1 → Eq (f_1 x) x
      f' : Equiv.Perm α
      hf' : Membership.mem l f'
      ⊢ Membership.mem (List.cons f l) f'
    -/
    simp only [List.find?, List.mem_cons]
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x : α
      f : Equiv.Perm α
      l : List (Equiv.Perm α)
      ih : (∀ (f : Equiv.Perm α), Membership.mem l f → Eq (f x) x) → Eq (l.prod x) x
      hx : ∀ (f_1 : Equiv.Perm α), Membership.mem (List.cons f l) f_1 → Eq (f_1 x) x
      f' : Equiv.Perm α
      hf' : Membership.mem l f'
      ⊢ Or (Eq f' f) (Membership.mem l f')
    -/
    exact Or.inr hf'
    /-
      🎉 no goals
    -/


theorem support_pow_le (σ : Perm α) (n : ℕ) : (σ ^ n).support ≤ σ.support := fun _ h1 =>
  mem_support.mpr fun h2 => mem_support.mp h1 (pow_apply_eq_self_of_apply_eq_self h2 n)


@[simp]
theorem support_inv (σ : Perm α) : support σ⁻¹ = σ.support := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    ⊢ Eq (Inv.inv σ).support σ.support
  -/
  simp_rw [Finset.ext_iff, mem_support, not_iff_not, inv_eq_iff_eq.trans eq_comm, imp_true_iff]
  /-
    🎉 no goals
  -/


theorem apply_mem_support {x : α} : f x ∈ f.support ↔ x ∈ f.support := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x : α
    ⊢ Iff (Membership.mem f.support (f x)) (Membership.mem f.support x)
  -/
  rw [mem_support, mem_support, Ne, Ne, apply_eq_iff_eq]
  /-
    🎉 no goals
  -/


/-- The support of a permutation is invariant -/
theorem isInvariant_of_support_le {c : Perm α} {s : Finset α} (hcs : c.support ≤ s) (x : α) :
    x ∈ s ↔ c x ∈ s := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    c : Equiv.Perm α
    s : Finset α
    hcs : LE.le c.support s
    x : α
    ⊢ Iff (Membership.mem s x) (Membership.mem s (c x))
  -/
  by_cases hx' : x ∈ c.support
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      c : Equiv.Perm α
      s : Finset α
      hcs : LE.le c.support s
      x : α
      hx' : Membership.mem c.support x
      ⊢ Iff (Membership.mem s x) (Membership.mem s (c x))
    -/
  · simp only [hcs hx', true_iff, hcs (apply_mem_support.mpr hx')]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      c : Equiv.Perm α
      s : Finset α
      hcs : LE.le c.support s
      x : α
      hx' : Not (Membership.mem c.support x)
      ⊢ Iff (Membership.mem s x) (Membership.mem s (c x))
    -/
  · rw [not_mem_support.mp hx']
    /-
      🎉 no goals
    -/


/-- A permutation c is the extension of a restriction of g to s
  iff its support is contained in s and its restriction is that of g -/
lemma ofSubtype_eq_iff {g c : Equiv.Perm α} {s : Finset α}
    (hg : ∀ x, x ∈ s ↔ g x ∈ s) :
    ofSubtype (g.subtypePerm hg) = c ↔
      c.support ≤ s ∧
      ∀ (hc' : ∀ x, x ∈ s ↔ c x ∈ s), c.subtypePerm hc' = g.subtypePerm hg := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    s : Finset α
    hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
    ⊢ Iff (Eq (Equiv.Perm.ofSubtype (g.subtypePerm hg)) c) (And (LE.le c.support s …
  -/
  simp only [Equiv.ext_iff, subtypePerm_apply, Subtype.mk.injEq, Subtype.forall]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    g c : Equiv.Perm α
    s : Finset α
    hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
    ⊢ Iff (∀ (x : α), Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) x) (c x)) (And …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g c : Equiv.Perm α
      s : Finset α
      hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
      ⊢ (∀ (x : α), Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) x) (c x)) → And (L …
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g c : Equiv.Perm α
      s : Finset α
      hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
      h : ∀ (x : α), Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) x) (c x)
      ⊢ And (LE.le c.support s) ((∀ (x : α), Iff (Membership.mem s x) (Membership.me …
    -/
    constructor
      /-
        case mp.left
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        g c : Equiv.Perm α
        s : Finset α
        hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
        h : ∀ (x : α), Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) x) (c x)
        ⊢ LE.le c.support s
      -/
    · intro a ha
      /-
        case mp.left
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        g c : Equiv.Perm α
        s : Finset α
        hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
        h : ∀ (x : α), Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) x) (c x)
        a : α
        ha : Membership.mem c.support a
        ⊢ Membership.mem s a
      -/
      by_contra ha'
      /-
        case mp.left
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        g c : Equiv.Perm α
        s : Finset α
        hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
        h : ∀ (x : α), Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) x) (c x)
        a : α
        ha : Membership.mem c.support a
        ha' : Not (Membership.mem s a)
        ⊢ False
      -/
      rw [mem_support, ← h a, ofSubtype_apply_of_not_mem (p := (· ∈ s)) _ ha'] at ha
      /-
        case mp.left
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        g c : Equiv.Perm α
        s : Finset α
        hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
        h : ∀ (x : α), Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) x) (c x)
        a : α
        ha : Ne a a
        ha' : Not (Membership.mem s a)
        ⊢ False
      -/
      exact ha rfl
      /-
        🎉 no goals
      -/
      /-
        case mp.right
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        g c : Equiv.Perm α
        s : Finset α
        hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
        h : ∀ (x : α), Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) x) (c x)
        ⊢ (∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (c x))) → ∀ (a : α),  …
      -/
    · intro _ a ha
      /-
        case mp.right
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        g c : Equiv.Perm α
        s : Finset α
        hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
        h : ∀ (x : α), Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) x) (c x)
        hc'✝ : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (c x))
        a : α
        ha : Membership.mem s a
        ⊢ Eq (c a) (g a)
      -/
      rw [← h a, ofSubtype_apply_of_mem (p := (· ∈ s)) _ ha, subtypePerm_apply]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g c : Equiv.Perm α
      s : Finset α
      hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
      ⊢ And (LE.le c.support s) ((∀ (x : α), Iff (Membership.mem s x) (Membership.me …
    -/
  · rintro ⟨hc, h⟩ a
    /-
      case mpr.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g c : Equiv.Perm α
      s : Finset α
      hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
      hc : LE.le c.support s
      h : (∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (c x))) → ∀ (a : α) …
      a : α
      ⊢ Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) a) (c a)
    -/
    specialize h (isInvariant_of_support_le hc)
    /-
      case mpr.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      g c : Equiv.Perm α
      s : Finset α
      hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
      hc : LE.le c.support s
      a : α
      h : ∀ (a : α), Membership.mem s a → Eq (c a) (g a)
      ⊢ Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) a) (c a)
    -/
    by_cases ha : a ∈ s
      /-
        case pos
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        g c : Equiv.Perm α
        s : Finset α
        hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
        hc : LE.le c.support s
        a : α
        h : ∀ (a : α), Membership.mem s a → Eq (c a) (g a)
        ha : Membership.mem s a
        ⊢ Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) a) (c a)
      -/
    · rw [h a ha, ofSubtype_apply_of_mem (p := (· ∈ s)) _ ha, subtypePerm_apply]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        g c : Equiv.Perm α
        s : Finset α
        hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
        hc : LE.le c.support s
        a : α
        h : ∀ (a : α), Membership.mem s a → Eq (c a) (g a)
        ha : Not (Membership.mem s a)
        ⊢ Eq ((Equiv.Perm.ofSubtype (g.subtypePerm hg)) a) (c a)
      -/
    · rw [ofSubtype_apply_of_not_mem (p := (· ∈ s)) _ ha, eq_comm, ← not_mem_support]
      /-
        case neg
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        g c : Equiv.Perm α
        s : Finset α
        hg : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (g x))
        hc : LE.le c.support s
        a : α
        h : ∀ (a : α), Membership.mem s a → Eq (c a) (g a)
        ha : Not (Membership.mem s a)
        ⊢ Not (Membership.mem c.support a)
      -/
      exact Finset.not_mem_mono hc ha
      /-
        🎉 no goals
      -/


theorem support_ofSubtype {p : α → Prop} [DecidablePred p] (u : Perm (Subtype p)) :
    (ofSubtype u).support = u.support.map (Function.Embedding.subtype p) := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    p : α → Prop
    inst✝ : DecidablePred p
    u : Equiv.Perm (Subtype p)
    ⊢ Eq (Equiv.Perm.ofSubtype u).support (Finset.map (Function.Embedding.subtype  …
  -/
  ext x
  simp only [mem_support, ne_eq, Finset.mem_map, Function.Embedding.coe_subtype, Subtype.exists,
    exists_and_right, exists_eq_right, not_iff_comm, not_exists, not_not]
  /-
    case h
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    p : α → Prop
    inst✝ : DecidablePred p
    u : Equiv.Perm (Subtype p)
    x : α
    ⊢ Iff (∀ (x_1 : p x), Eq (u ⟨x, ⋯⟩) ⟨x, ⋯⟩) (Eq ((Equiv.Perm.ofSubtype u) x) x)
  -/
  by_cases hx : p x
    /-
      case pos
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      p : α → Prop
      inst✝ : DecidablePred p
      u : Equiv.Perm (Subtype p)
      x : α
      hx : p x
      ⊢ Iff (∀ (x_1 : p x), Eq (u ⟨x, ⋯⟩) ⟨x, ⋯⟩) (Eq ((Equiv.Perm.ofSubtype u) x) x)
    -/
  · simp only [forall_prop_of_true hx, ofSubtype_apply_of_mem u hx, ← Subtype.coe_inj]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      p : α → Prop
      inst✝ : DecidablePred p
      u : Equiv.Perm (Subtype p)
      x : α
      hx : Not (p x)
      ⊢ Iff (∀ (x_1 : p x), Eq (u ⟨x, ⋯⟩) ⟨x, ⋯⟩) (Eq ((Equiv.Perm.ofSubtype u) x) x)
    -/
  · simp only [forall_prop_of_false hx, true_iff, ofSubtype_apply_of_not_mem u hx]
    /-
      🎉 no goals
    -/


theorem pow_apply_mem_support {n : ℕ} {x : α} : (f ^ n) x ∈ f.support ↔ x ∈ f.support := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    n : Nat
    x : α
    ⊢ Iff (Membership.mem f.support ((HPow.hPow f n) x)) (Membership.mem f.support …
  -/
  simp only [mem_support, ne_eq, apply_pow_apply_eq_iff]
  /-
    🎉 no goals
  -/


theorem zpow_apply_mem_support {n : ℤ} {x : α} : (f ^ n) x ∈ f.support ↔ x ∈ f.support := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    n : Int
    x : α
    ⊢ Iff (Membership.mem f.support ((HPow.hPow f n) x)) (Membership.mem f.support …
  -/
  simp only [mem_support, ne_eq, apply_zpow_apply_eq_iff]
  /-
    🎉 no goals
  -/


theorem pow_eq_on_of_mem_support (h : ∀ x ∈ f.support ∩ g.support, f x = g x) (k : ℕ) :
    ∀ x ∈ f.support ∩ g.support, (f ^ k) x = (g ^ k) x := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq (f x) ( …
    k : Nat
    ⊢ ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq ((HPow.hP …
  -/
  induction' k with k hk
    /-
      case zero
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq (f x) ( …
      ⊢ ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq ((HPow.hP …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq (f x) ( …
      k : Nat
      hk : ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq ((HPow …
      ⊢ ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq ((HPow.hP …
    -/
  · intro x hx
    /-
      case succ
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq (f x) ( …
      k : Nat
      hk : ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq ((HPow …
      x : α
      hx : Membership.mem (Inter.inter f.support g.support) x
      ⊢ Eq ((HPow.hPow f (HAdd.hAdd k 1)) x) ((HPow.hPow g (HAdd.hAdd k 1)) x)
    -/
    rw [pow_succ, mul_apply, pow_succ, mul_apply, h _ hx, hk]
    /-
      case succ.a
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq (f x) ( …
      k : Nat
      hk : ∀ (x : α), Membership.mem (Inter.inter f.support g.support) x → Eq ((HPow …
      x : α
      hx : Membership.mem (Inter.inter f.support g.support) x
      ⊢ Membership.mem (Inter.inter f.support g.support) (g x)
    -/
    rwa [mem_inter, apply_mem_support, ← h _ hx, apply_mem_support, ← mem_inter]
    /-
      🎉 no goals
    -/


theorem disjoint_iff_disjoint_support : Disjoint f g ↔ _root_.Disjoint f.support g.support := by
  simp [disjoint_iff_eq_or_eq, disjoint_iff, disjoint_iff, Finset.ext_iff, not_and_or,
    imp_iff_not_or]


theorem Disjoint.disjoint_support (h : Disjoint f g) : _root_.Disjoint f.support g.support :=
  disjoint_iff_disjoint_support.1 h


theorem Disjoint.support_mul (h : Disjoint f g) : (f * g).support = f.support ∪ g.support := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : f.Disjoint g
    ⊢ Eq (HMul.hMul f g).support (Union.union f.support g.support)
  -/
  refine le_antisymm (support_mul_le _ _) fun a => ?_
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : f.Disjoint g
    a : α
    ⊢ Membership.mem (Union.union f.support g.support) a → Membership.mem (HMul.hM …
  -/
  rw [mem_union, mem_support, mem_support, mem_support, mul_apply, ← not_and_or, not_imp_not]
  exact
    (h a).elim (fun hf h => ⟨hf, f.apply_eq_iff_eq.mp (h.trans hf.symm)⟩) fun hg h =>
      ⟨(congr_arg f hg).symm.trans h, hg⟩


theorem support_prod_of_pairwise_disjoint (l : List (Perm α)) (h : l.Pairwise Disjoint) :
    l.prod.support = (l.map support).foldr (· ⊔ ·) ⊥ := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l : List (Equiv.Perm α)
    h : List.Pairwise Equiv.Perm.Disjoint l
    ⊢ Eq l.prod.support (List.foldr (fun x1 x2 => Max.max x1 x2) Bot.bot (List.map …
  -/
  induction' l with hd tl hl
    /-
      case nil
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      h : List.Pairwise Equiv.Perm.Disjoint List.nil
      ⊢ Eq List.nil.prod.support (List.foldr (fun x1 x2 => Max.max x1 x2) Bot.bot (L …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hd : Equiv.Perm α
      tl : List (Equiv.Perm α)
      hl : List.Pairwise Equiv.Perm.Disjoint tl → Eq tl.prod.support (List.foldr (fu …
      h : List.Pairwise Equiv.Perm.Disjoint (List.cons hd tl)
      ⊢ Eq (List.cons hd tl).prod.support (List.foldr (fun x1 x2 => Max.max x1 x2) B …
    -/
  · rw [List.pairwise_cons] at h
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hd : Equiv.Perm α
      tl : List (Equiv.Perm α)
      hl : List.Pairwise Equiv.Perm.Disjoint tl → Eq tl.prod.support (List.foldr (fu …
      h : And (∀ (a' : Equiv.Perm α), Membership.mem tl a' → hd.Disjoint a') (List.P …
      ⊢ Eq (List.cons hd tl).prod.support (List.foldr (fun x1 x2 => Max.max x1 x2) B …
    -/
    have : Disjoint hd tl.prod := disjoint_prod_right _ h.left
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hd : Equiv.Perm α
      tl : List (Equiv.Perm α)
      hl : List.Pairwise Equiv.Perm.Disjoint tl → Eq tl.prod.support (List.foldr (fu …
      h : And (∀ (a' : Equiv.Perm α), Membership.mem tl a' → hd.Disjoint a') (List.P …
      this : hd.Disjoint tl.prod
      ⊢ Eq (List.cons hd tl).prod.support (List.foldr (fun x1 x2 => Max.max x1 x2) B …
    -/
    simp [this.support_mul, hl h.right]
    /-
      🎉 no goals
    -/


theorem support_noncommProd {ι : Type*} {k : ι → Perm α} {s : Finset ι}
    (hs : Set.Pairwise s fun i j ↦ Disjoint (k i) (k j)) :
    (s.noncommProd k (hs.imp (fun _ _ ↦ Perm.Disjoint.commute))).support =
      s.biUnion fun i ↦ (k i).support := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | @insert i s hi hrec =>
    have hs' : (s : Set ι).Pairwise fun i j ↦ Disjoint (k i) (k j) :=
      hs.mono (by simp only [Finset.coe_insert, Set.subset_insert])
    rw [Finset.noncommProd_insert_of_not_mem _ _ _ _ hi, Finset.biUnion_insert]
    rw [Equiv.Perm.Disjoint.support_mul, hrec hs']
    apply disjoint_noncommProd_right
    intro j hj
    apply hs _ _ (ne_of_mem_of_not_mem hj hi).symm <;>
      simp only [Finset.coe_insert, Set.mem_insert_iff, Finset.mem_coe, hj, or_true, true_or]


theorem support_prod_le (l : List (Perm α)) : l.prod.support ≤ (l.map support).foldr (· ⊔ ·) ⊥ := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l : List (Equiv.Perm α)
    ⊢ LE.le l.prod.support (List.foldr (fun x1 x2 => Max.max x1 x2) Bot.bot (List. …
  -/
  induction' l with hd tl hl
    /-
      case nil
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      ⊢ LE.le List.nil.prod.support (List.foldr (fun x1 x2 => Max.max x1 x2) Bot.bot …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hd : Equiv.Perm α
      tl : List (Equiv.Perm α)
      hl : LE.le tl.prod.support (List.foldr (fun x1 x2 => Max.max x1 x2) Bot.bot (L …
      ⊢ LE.le (List.cons hd tl).prod.support (List.foldr (fun x1 x2 => Max.max x1 x2 …
    -/
  · rw [List.prod_cons, List.map_cons, List.foldr_cons]
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hd : Equiv.Perm α
      tl : List (Equiv.Perm α)
      hl : LE.le tl.prod.support (List.foldr (fun x1 x2 => Max.max x1 x2) Bot.bot (L …
      ⊢ LE.le (HMul.hMul hd tl.prod).support (Max.max hd.support (List.foldr (fun x1 …
    -/
    refine (support_mul_le hd tl.prod).trans ?_
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hd : Equiv.Perm α
      tl : List (Equiv.Perm α)
      hl : LE.le tl.prod.support (List.foldr (fun x1 x2 => Max.max x1 x2) Bot.bot (L …
      ⊢ LE.le (Max.max hd.support tl.prod.support) (Max.max hd.support (List.foldr ( …
    -/
    exact sup_le_sup le_rfl hl
    /-
      🎉 no goals
    -/


theorem support_zpow_le (σ : Perm α) (n : ℤ) : (σ ^ n).support ≤ σ.support := fun _ h1 =>
  mem_support.mpr fun h2 => mem_support.mp h1 (zpow_apply_eq_self_of_apply_eq_self h2 n)


@[simp]
theorem support_swap {x y : α} (h : x ≠ y) : support (swap x y) = {x, y} := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x y : α
    h : Ne x y
    ⊢ Eq (Equiv.swap x y).support (Insert.insert x (Singleton.singleton y))
  -/
  ext z
  /-
    case h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x y : α
    h : Ne x y
    z : α
    ⊢ Iff (Membership.mem (Equiv.swap x y).support z) (Membership.mem (Insert.inse …
  -/
  by_cases hx : z = x
  /-
    case pos
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x y : α
    h : Ne x y
    z : α
    hx : Eq z x
    ⊢ Iff (Membership.mem (Equiv.swap x y).support z) (Membership.mem (Insert.inse …
  -/
  any_goals simpa [hx] using h.symm
  /-
    case neg
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x y : α
    h : Ne x y
    z : α
    hx : Not (Eq z x)
    ⊢ Iff (Membership.mem (Equiv.swap x y).support z) (Membership.mem (Insert.inse …
  -/
  by_cases hy : z = y
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x y : α
      h : Ne x y
      z : α
      hx : Not (Eq z x)
      hy : Eq z y
      ⊢ Iff (Membership.mem (Equiv.swap x y).support z) (Membership.mem (Insert.inse …
    -/
  · simpa [swap_apply_of_ne_of_ne, hx, hy] using h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x y : α
      h : Ne x y
      z : α
      hx : Not (Eq z x)
      hy : Not (Eq z y)
      ⊢ Iff (Membership.mem (Equiv.swap x y).support z) (Membership.mem (Insert.inse …
    -/
  · simp [swap_apply_of_ne_of_ne, hx, hy]
    /-
      🎉 no goals
    -/


theorem support_swap_iff (x y : α) : support (swap x y) = {x, y} ↔ x ≠ y := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x y : α
    ⊢ Iff (Eq (Equiv.swap x y).support (Insert.insert x (Singleton.singleton y)))  …
  -/
  refine ⟨fun h => ?_, fun h => support_swap h⟩
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x y : α
    h : Eq (Equiv.swap x y).support (Insert.insert x (Singleton.singleton y))
    ⊢ Ne x y
  -/
  rintro rfl
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x : α
    h : Eq (Equiv.swap x x).support (Insert.insert x (Singleton.singleton x))
    ⊢ False
  -/
  simp [Finset.ext_iff] at h
  /-
    🎉 no goals
  -/


theorem support_swap_mul_swap {x y z : α} (h : List.Nodup [x, y, z]) :
    support (swap x y * swap y z) = {x, y, z} := by
  simp only [List.not_mem_nil, and_true, List.mem_cons, not_false_iff, List.nodup_cons,
    List.mem_singleton, and_self_iff, List.nodup_nil] at h
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x y z : α
    h : And (Not (Or (Eq x y) (Or (Eq x z) False))) (Not (Or (Eq y z) False))
    ⊢ Eq (HMul.hMul (Equiv.swap x y) (Equiv.swap y z)).support (Insert.insert x (I …
  -/
  push_neg at h
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x y z : α
    h : And (And (Ne x y) (And (Ne x z) (Not False))) (And (Ne y z) (Not False))
    ⊢ Eq (HMul.hMul (Equiv.swap x y) (Equiv.swap y z)).support (Insert.insert x (I …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x y z : α
      h : And (And (Ne x y) (And (Ne x z) (Not False))) (And (Ne y z) (Not False))
      ⊢ LE.le (HMul.hMul (Equiv.swap x y) (Equiv.swap y z)).support (Insert.insert x …
    -/
  · convert support_mul_le (swap x y) (swap y z) using 1
    /-
      case h.e'_4
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x y z : α
      h : And (And (Ne x y) (And (Ne x z) (Not False))) (And (Ne y z) (Not False))
      ⊢ Eq (Insert.insert x (Insert.insert y (Singleton.singleton z))) (Max.max (Equ …
    -/
    rw [support_swap h.left.left, support_swap h.right.left]
    /-
      case h.e'_4
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x y z : α
      h : And (And (Ne x y) (And (Ne x z) (Not False))) (And (Ne y z) (Not False))
      ⊢ Eq (Insert.insert x (Insert.insert y (Singleton.singleton z))) (Max.max (Ins …
    -/
    simp [Finset.ext_iff]
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x y z : α
      h : And (And (Ne x y) (And (Ne x z) (Not False))) (And (Ne y z) (Not False))
      ⊢ LE.le (Insert.insert x (Insert.insert y (Singleton.singleton z))) (HMul.hMul …
    -/
  · intro
    /-
      case a
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x y z : α
      h : And (And (Ne x y) (And (Ne x z) (Not False))) (And (Ne y z) (Not False))
      a✝ : α
      ⊢ Membership.mem (Insert.insert x (Insert.insert y (Singleton.singleton z))) a …
    -/
    simp only [mem_insert, mem_singleton]
    /-
      case a
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x y z : α
      h : And (And (Ne x y) (And (Ne x z) (Not False))) (And (Ne y z) (Not False))
      a✝ : α
      ⊢ Or (Eq a✝ x) (Or (Eq a✝ y) (Eq a✝ z)) → Membership.mem (HMul.hMul (Equiv.swa …
    -/
    rintro (rfl | rfl | rfl | _) <;>
      simp [swap_apply_of_ne_of_ne, h.left.left, h.left.left.symm, h.left.right.symm,
        h.left.right.left.symm, h.right.left.symm]


theorem support_swap_mul_ge_support_diff (f : Perm α) (x y : α) :
    f.support \ {x, y} ≤ (swap x y * f).support := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x y : α
    ⊢ LE.le (SDiff.sdiff f.support (Insert.insert x (Singleton.singleton y))) (HMu …
  -/
  intro
  simp only [and_imp, Perm.coe_mul, Function.comp_apply, Ne, mem_support, mem_insert, mem_sdiff,
    mem_singleton]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x y a✝ : α
    ⊢ Not (Eq (f a✝) a✝) → Not (Or (Eq a✝ x) (Eq a✝ y)) → Not (Eq ((Equiv.swap x y …
  -/
  push_neg
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x y a✝ : α
    ⊢ Ne (f a✝) a✝ → And (Ne a✝ x) (Ne a✝ y) → Ne ((Equiv.swap x y) (f a✝)) a✝
  -/
  rintro ha ⟨hx, hy⟩ H
  /-
    case intro
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x y a✝ : α
    ha : Ne (f a✝) a✝
    hx : Ne a✝ x
    hy : Ne a✝ y
    H : Eq ((Equiv.swap x y) (f a✝)) a✝
    ⊢ False
  -/
  rw [swap_apply_eq_iff, swap_apply_of_ne_of_ne hx hy] at H
  /-
    case intro
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x y a✝ : α
    ha : Ne (f a✝) a✝
    hx : Ne a✝ x
    hy : Ne a✝ y
    H : Eq (f a✝) a✝
    ⊢ False
  -/
  exact ha H
  /-
    🎉 no goals
  -/


theorem support_swap_mul_eq (f : Perm α) (x : α) (h : f (f x) ≠ x) :
    (swap x (f x) * f).support = f.support \ {x} := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x : α
    h : Ne (f (f x)) x
    ⊢ Eq (HMul.hMul (Equiv.swap x (f x)) f).support (SDiff.sdiff f.support (Single …
  -/
  by_cases hx : f x = x
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      h : Ne (f (f x)) x
      hx : Eq (f x) x
      ⊢ Eq (HMul.hMul (Equiv.swap x (f x)) f).support (SDiff.sdiff f.support (Single …
    -/
  · simp [hx, sdiff_singleton_eq_erase, not_mem_support.mpr hx, erase_eq_of_not_mem]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x : α
    h : Ne (f (f x)) x
    hx : Not (Eq (f x) x)
    ⊢ Eq (HMul.hMul (Equiv.swap x (f x)) f).support (SDiff.sdiff f.support (Single …
  -/
  ext z
  /-
    case neg.h
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x : α
    h : Ne (f (f x)) x
    hx : Not (Eq (f x) x)
    z : α
    ⊢ Iff (Membership.mem (HMul.hMul (Equiv.swap x (f x)) f).support z) (Membershi …
  -/
  by_cases hzx : z = x
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      h : Ne (f (f x)) x
      hx : Not (Eq (f x) x)
      z : α
      hzx : Eq z x
      ⊢ Iff (Membership.mem (HMul.hMul (Equiv.swap x (f x)) f).support z) (Membershi …
    -/
  · simp [hzx]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x : α
    h : Ne (f (f x)) x
    hx : Not (Eq (f x) x)
    z : α
    hzx : Not (Eq z x)
    ⊢ Iff (Membership.mem (HMul.hMul (Equiv.swap x (f x)) f).support z) (Membershi …
  -/
  by_cases hzf : z = f x
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      h : Ne (f (f x)) x
      hx : Not (Eq (f x) x)
      z : α
      hzx : Not (Eq z x)
      hzf : Eq z (f x)
      ⊢ Iff (Membership.mem (HMul.hMul (Equiv.swap x (f x)) f).support z) (Membershi …
    -/
  · simp [hzf, hx, h, swap_apply_of_ne_of_ne]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x : α
    h : Ne (f (f x)) x
    hx : Not (Eq (f x) x)
    z : α
    hzx : Not (Eq z x)
    hzf : Not (Eq z (f x))
    ⊢ Iff (Membership.mem (HMul.hMul (Equiv.swap x (f x)) f).support z) (Membershi …
  -/
  by_cases hzfx : f z = x
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      h : Ne (f (f x)) x
      hx : Not (Eq (f x) x)
      z : α
      hzx : Not (Eq z x)
      hzf : Not (Eq z (f x))
      hzfx : Eq (f z) x
      ⊢ Iff (Membership.mem (HMul.hMul (Equiv.swap x (f x)) f).support z) (Membershi …
    -/
  · simp [Ne.symm hzx, hzx, Ne.symm hzf, hzfx]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x : α
      h : Ne (f (f x)) x
      hx : Not (Eq (f x) x)
      z : α
      hzx : Not (Eq z x)
      hzf : Not (Eq z (f x))
      hzfx : Not (Eq (f z) x)
      ⊢ Iff (Membership.mem (HMul.hMul (Equiv.swap x (f x)) f).support z) (Membershi …
    -/
  · simp [Ne.symm hzx, hzx, Ne.symm hzf, hzfx, f.injective.ne hzx, swap_apply_of_ne_of_ne]
    /-
      🎉 no goals
    -/


theorem mem_support_swap_mul_imp_mem_support_ne {x y : α} (hy : y ∈ support (swap x (f x) * f)) :
    y ∈ support f ∧ y ≠ x := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x y : α
    hy : Membership.mem (HMul.hMul (Equiv.swap x (f x)) f).support y
    ⊢ And (Membership.mem f.support y) (Ne y x)
  -/
  simp only [mem_support, swap_apply_def, mul_apply, f.injective.eq_iff] at *
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    x y : α
    hy : Ne (ite (Eq (f y) x) (f x) (ite (Eq y x) x (f y))) y
    ⊢ And (Ne (f y) y) (Ne y x)
  -/
  by_cases h : f y = x
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x y : α
      hy : Ne (ite (Eq (f y) x) (f x) (ite (Eq y x) x (f y))) y
      h : Eq (f y) x
      ⊢ And (Ne (f y) y) (Ne y x)
    -/
                              /-
                                🎉 no goals
                              -/
  · constructor <;> intro <;> simp_all only [if_true, eq_self_iff_true, not_true, Ne]
                              /-
                                🎉 no goals
                              -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      x y : α
      hy : Ne (ite (Eq (f y) x) (f x) (ite (Eq y x) x (f y))) y
      h : Not (Eq (f y) x)
      ⊢ And (Ne (f y) y) (Ne y x)
    -/
  · split_ifs at hy with heq
      /-
        case pos
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f : Equiv.Perm α
        x y : α
        h : Not (Eq (f y) x)
        heq : Eq y x
        hy : Ne x y
        ⊢ And (Ne (f y) y) (Ne y x)
      -/
    · subst heq; exact ⟨h, hy⟩
                 /-
                   🎉 no goals
                 -/
      /-
        case neg
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f : Equiv.Perm α
        x y : α
        h : Not (Eq (f y) x)
        heq : Not (Eq y x)
        hy : Ne (f y) y
        ⊢ And (Ne (f y) y) (Ne y x)
      -/
    · exact ⟨hy, heq⟩
      /-
        🎉 no goals
      -/


theorem Disjoint.mem_imp (h : Disjoint f g) {x : α} (hx : x ∈ f.support) : x ∉ g.support :=
  disjoint_left.mp h.disjoint_support hx


theorem eq_on_support_mem_disjoint {l : List (Perm α)} (h : f ∈ l) (hl : l.Pairwise Disjoint) :
    ∀ x ∈ f.support, f x = l.prod x := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    l : List (Equiv.Perm α)
    h : Membership.mem l f
    hl : List.Pairwise Equiv.Perm.Disjoint l
    ⊢ ∀ (x : α), Membership.mem f.support x → Eq (f x) (l.prod x)
  -/
  induction' l with hd tl IH
    /-
      case nil
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : Membership.mem List.nil f
      hl : List.Pairwise Equiv.Perm.Disjoint List.nil
      ⊢ ∀ (x : α), Membership.mem f.support x → Eq (f x) (List.nil.prod x)
    -/
  · simp at h
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f hd : Equiv.Perm α
      tl : List (Equiv.Perm α)
      IH : Membership.mem tl f → List.Pairwise Equiv.Perm.Disjoint tl → ∀ (x : α), M …
      h : Membership.mem (List.cons hd tl) f
      hl : List.Pairwise Equiv.Perm.Disjoint (List.cons hd tl)
      ⊢ ∀ (x : α), Membership.mem f.support x → Eq (f x) ((List.cons hd tl).prod x)
    -/
  · intro x hx
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f hd : Equiv.Perm α
      tl : List (Equiv.Perm α)
      IH : Membership.mem tl f → List.Pairwise Equiv.Perm.Disjoint tl → ∀ (x : α), M …
      h : Membership.mem (List.cons hd tl) f
      hl : List.Pairwise Equiv.Perm.Disjoint (List.cons hd tl)
      x : α
      hx : Membership.mem f.support x
      ⊢ Eq (f x) ((List.cons hd tl).prod x)
    -/
    rw [List.pairwise_cons] at hl
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f hd : Equiv.Perm α
      tl : List (Equiv.Perm α)
      IH : Membership.mem tl f → List.Pairwise Equiv.Perm.Disjoint tl → ∀ (x : α), M …
      h : Membership.mem (List.cons hd tl) f
      hl : And (∀ (a' : Equiv.Perm α), Membership.mem tl a' → hd.Disjoint a') (List. …
      x : α
      hx : Membership.mem f.support x
      ⊢ Eq (f x) ((List.cons hd tl).prod x)
    -/
    rw [List.mem_cons] at h
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f hd : Equiv.Perm α
      tl : List (Equiv.Perm α)
      IH : Membership.mem tl f → List.Pairwise Equiv.Perm.Disjoint tl → ∀ (x : α), M …
      h : Or (Eq f hd) (Membership.mem tl f)
      hl : And (∀ (a' : Equiv.Perm α), Membership.mem tl a' → hd.Disjoint a') (List. …
      x : α
      hx : Membership.mem f.support x
      ⊢ Eq (f x) ((List.cons hd tl).prod x)
    -/
    rcases h with (rfl | h)
    · rw [List.prod_cons, mul_apply,
        not_mem_support.mp ((disjoint_prod_right tl hl.left).mem_imp hx)]
      /-
        case cons.inr
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f hd : Equiv.Perm α
        tl : List (Equiv.Perm α)
        IH : Membership.mem tl f → List.Pairwise Equiv.Perm.Disjoint tl → ∀ (x : α), M …
        hl : And (∀ (a' : Equiv.Perm α), Membership.mem tl a' → hd.Disjoint a') (List. …
        x : α
        hx : Membership.mem f.support x
        h : Membership.mem tl f
        ⊢ Eq (f x) ((List.cons hd tl).prod x)
      -/
    · rw [List.prod_cons, mul_apply, ← IH h hl.right _ hx, eq_comm, ← not_mem_support]
      /-
        case cons.inr
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f hd : Equiv.Perm α
        tl : List (Equiv.Perm α)
        IH : Membership.mem tl f → List.Pairwise Equiv.Perm.Disjoint tl → ∀ (x : α), M …
        hl : And (∀ (a' : Equiv.Perm α), Membership.mem tl a' → hd.Disjoint a') (List. …
        x : α
        hx : Membership.mem f.support x
        h : Membership.mem tl f
        ⊢ Not (Membership.mem hd.support (f x))
      -/
      refine (hl.left _ h).symm.mem_imp ?_
      /-
        case cons.inr
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f hd : Equiv.Perm α
        tl : List (Equiv.Perm α)
        IH : Membership.mem tl f → List.Pairwise Equiv.Perm.Disjoint tl → ∀ (x : α), M …
        hl : And (∀ (a' : Equiv.Perm α), Membership.mem tl a' → hd.Disjoint a') (List. …
        x : α
        hx : Membership.mem f.support x
        h : Membership.mem tl f
        ⊢ Membership.mem f.support (f x)
      -/
      simpa using hx
      /-
        🎉 no goals
      -/


theorem Disjoint.mono {x y : Perm α} (h : Disjoint f g) (hf : x.support ≤ f.support)
    (hg : y.support ≤ g.support) : Disjoint x y := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g x y : Equiv.Perm α
    h : f.Disjoint g
    hf : LE.le x.support f.support
    hg : LE.le y.support g.support
    ⊢ x.Disjoint y
  -/
  rw [disjoint_iff_disjoint_support] at h ⊢
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g x y : Equiv.Perm α
    h : _root_.Disjoint f.support g.support
    hf : LE.le x.support f.support
    hg : LE.le y.support g.support
    ⊢ _root_.Disjoint x.support y.support
  -/
  exact h.mono hf hg
  /-
    🎉 no goals
  -/


theorem support_le_prod_of_mem {l : List (Perm α)} (h : f ∈ l) (hl : l.Pairwise Disjoint) :
    f.support ≤ l.prod.support := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    l : List (Equiv.Perm α)
    h : Membership.mem l f
    hl : List.Pairwise Equiv.Perm.Disjoint l
    ⊢ LE.le f.support l.prod.support
  -/
  intro x hx
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    l : List (Equiv.Perm α)
    h : Membership.mem l f
    hl : List.Pairwise Equiv.Perm.Disjoint l
    x : α
    hx : Membership.mem f.support x
    ⊢ Membership.mem l.prod.support x
  -/
  rwa [mem_support, ← eq_on_support_mem_disjoint h hl _ hx, ← mem_support]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_extend_domain (f : α ≃ Subtype p) {g : Perm α} :
    support (g.extendDomain f) = g.support.map f.asEmbedding := by
  /-
    α : Type u_1
    inst✝⁴ : DecidableEq α
    inst✝³ : Fintype α
    β : Type u_2
    inst✝² : DecidableEq β
    inst✝¹ : Fintype β
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    g : Equiv.Perm α
    ⊢ Eq (g.extendDomain f).support (Finset.map f.asEmbedding g.support)
  -/
  ext b
  simp only [exists_prop, Function.Embedding.coeFn_mk, toEmbedding_apply, mem_map, Ne,
    Function.Embedding.trans_apply, mem_support]
  /-
    case h
    α : Type u_1
    inst✝⁴ : DecidableEq α
    inst✝³ : Fintype α
    β : Type u_2
    inst✝² : DecidableEq β
    inst✝¹ : Fintype β
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    g : Equiv.Perm α
    b : β
    ⊢ Iff (Not (Eq ((g.extendDomain f) b) b)) (Exists fun a => And (Not (Eq (g a)  …
  -/
  by_cases pb : p b
    /-
      case pos
      α : Type u_1
      inst✝⁴ : DecidableEq α
      inst✝³ : Fintype α
      β : Type u_2
      inst✝² : DecidableEq β
      inst✝¹ : Fintype β
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      g : Equiv.Perm α
      b : β
      pb : p b
      ⊢ Iff (Not (Eq ((g.extendDomain f) b) b)) (Exists fun a => And (Not (Eq (g a)  …
    -/
  · rw [extendDomain_apply_subtype _ _ pb]
    /-
      case pos
      α : Type u_1
      inst✝⁴ : DecidableEq α
      inst✝³ : Fintype α
      β : Type u_2
      inst✝² : DecidableEq β
      inst✝¹ : Fintype β
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      g : Equiv.Perm α
      b : β
      pb : p b
      ⊢ Iff (Not (Eq (↑(f (g (f.symm ⟨b, pb⟩)))) b)) (Exists fun a => And (Not (Eq ( …
    -/
    constructor
      /-
        case pos.mp
        α : Type u_1
        inst✝⁴ : DecidableEq α
        inst✝³ : Fintype α
        β : Type u_2
        inst✝² : DecidableEq β
        inst✝¹ : Fintype β
        p : β → Prop
        inst✝ : DecidablePred p
        f : Equiv α (Subtype p)
        g : Equiv.Perm α
        b : β
        pb : p b
        ⊢ Not (Eq (↑(f (g (f.symm ⟨b, pb⟩)))) b) → Exists fun a => And (Not (Eq (g a)  …
      -/
    · rintro h
      /-
        case pos.mp
        α : Type u_1
        inst✝⁴ : DecidableEq α
        inst✝³ : Fintype α
        β : Type u_2
        inst✝² : DecidableEq β
        inst✝¹ : Fintype β
        p : β → Prop
        inst✝ : DecidablePred p
        f : Equiv α (Subtype p)
        g : Equiv.Perm α
        b : β
        pb : p b
        h : Not (Eq (↑(f (g (f.symm ⟨b, pb⟩)))) b)
        ⊢ Exists fun a => And (Not (Eq (g a) a)) (Eq (f.asEmbedding a) b)
      -/
      refine ⟨f.symm ⟨b, pb⟩, ?_, by simp⟩
      /-
        case pos.mp
        α : Type u_1
        inst✝⁴ : DecidableEq α
        inst✝³ : Fintype α
        β : Type u_2
        inst✝² : DecidableEq β
        inst✝¹ : Fintype β
        p : β → Prop
        inst✝ : DecidablePred p
        f : Equiv α (Subtype p)
        g : Equiv.Perm α
        b : β
        pb : p b
        h : Not (Eq (↑(f (g (f.symm ⟨b, pb⟩)))) b)
        ⊢ Not (Eq (g (f.symm ⟨b, pb⟩)) (f.symm ⟨b, pb⟩))
      -/
      contrapose! h
      /-
        case pos.mp
        α : Type u_1
        inst✝⁴ : DecidableEq α
        inst✝³ : Fintype α
        β : Type u_2
        inst✝² : DecidableEq β
        inst✝¹ : Fintype β
        p : β → Prop
        inst✝ : DecidablePred p
        f : Equiv α (Subtype p)
        g : Equiv.Perm α
        b : β
        pb : p b
        h : Eq (g (f.symm ⟨b, pb⟩)) (f.symm ⟨b, pb⟩)
        ⊢ Eq (↑(f (g (f.symm ⟨b, pb⟩)))) b
      -/
      simp [h]
      /-
        🎉 no goals
      -/
      /-
        case pos.mpr
        α : Type u_1
        inst✝⁴ : DecidableEq α
        inst✝³ : Fintype α
        β : Type u_2
        inst✝² : DecidableEq β
        inst✝¹ : Fintype β
        p : β → Prop
        inst✝ : DecidablePred p
        f : Equiv α (Subtype p)
        g : Equiv.Perm α
        b : β
        pb : p b
        ⊢ (Exists fun a => And (Not (Eq (g a) a)) (Eq (f.asEmbedding a) b)) → Not (Eq  …
      -/
    · rintro ⟨a, ha, hb⟩
      /-
        case pos.mpr.intro.intro
        α : Type u_1
        inst✝⁴ : DecidableEq α
        inst✝³ : Fintype α
        β : Type u_2
        inst✝² : DecidableEq β
        inst✝¹ : Fintype β
        p : β → Prop
        inst✝ : DecidablePred p
        f : Equiv α (Subtype p)
        g : Equiv.Perm α
        b : β
        pb : p b
        a : α
        ha : Not (Eq (g a) a)
        hb : Eq (f.asEmbedding a) b
        ⊢ Not (Eq (↑(f (g (f.symm ⟨b, pb⟩)))) b)
      -/
      contrapose! ha
      obtain rfl : a = f.symm ⟨b, pb⟩ := by
        rw [eq_symm_apply]
        exact Subtype.coe_injective hb
      /-
        case pos.mpr.intro.intro
        α : Type u_1
        inst✝⁴ : DecidableEq α
        inst✝³ : Fintype α
        β : Type u_2
        inst✝² : DecidableEq β
        inst✝¹ : Fintype β
        p : β → Prop
        inst✝ : DecidablePred p
        f : Equiv α (Subtype p)
        g : Equiv.Perm α
        b : β
        pb : p b
        ha : Eq (↑(f (g (f.symm ⟨b, pb⟩)))) b
        hb : Eq (f.asEmbedding (f.symm ⟨b, pb⟩)) b
        ⊢ Eq (g (f.symm ⟨b, pb⟩)) (f.symm ⟨b, pb⟩)
      -/
      rw [eq_symm_apply]
      /-
        case pos.mpr.intro.intro
        α : Type u_1
        inst✝⁴ : DecidableEq α
        inst✝³ : Fintype α
        β : Type u_2
        inst✝² : DecidableEq β
        inst✝¹ : Fintype β
        p : β → Prop
        inst✝ : DecidablePred p
        f : Equiv α (Subtype p)
        g : Equiv.Perm α
        b : β
        pb : p b
        ha : Eq (↑(f (g (f.symm ⟨b, pb⟩)))) b
        hb : Eq (f.asEmbedding (f.symm ⟨b, pb⟩)) b
        ⊢ Eq (f (g (f.symm ⟨b, pb⟩))) ⟨b, pb⟩
      -/
      exact Subtype.coe_injective ha
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝⁴ : DecidableEq α
      inst✝³ : Fintype α
      β : Type u_2
      inst✝² : DecidableEq β
      inst✝¹ : Fintype β
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      g : Equiv.Perm α
      b : β
      pb : Not (p b)
      ⊢ Iff (Not (Eq ((g.extendDomain f) b) b)) (Exists fun a => And (Not (Eq (g a)  …
    -/
  · rw [extendDomain_apply_not_subtype _ _ pb]
    /-
      case neg
      α : Type u_1
      inst✝⁴ : DecidableEq α
      inst✝³ : Fintype α
      β : Type u_2
      inst✝² : DecidableEq β
      inst✝¹ : Fintype β
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      g : Equiv.Perm α
      b : β
      pb : Not (p b)
      ⊢ Iff (Not (Eq b b)) (Exists fun a => And (Not (Eq (g a) a)) (Eq (f.asEmbeddin …
    -/
    simp only [not_exists, false_iff, not_and, eq_self_iff_true, not_true]
    /-
      case neg
      α : Type u_1
      inst✝⁴ : DecidableEq α
      inst✝³ : Fintype α
      β : Type u_2
      inst✝² : DecidableEq β
      inst✝¹ : Fintype β
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      g : Equiv.Perm α
      b : β
      pb : Not (p b)
      ⊢ ∀ (x : α), Not (Eq (g x) x) → Not (Eq (f.asEmbedding x) b)
    -/
    rintro a _ rfl
    /-
      case neg
      α : Type u_1
      inst✝⁴ : DecidableEq α
      inst✝³ : Fintype α
      β : Type u_2
      inst✝² : DecidableEq β
      inst✝¹ : Fintype β
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      g : Equiv.Perm α
      a : α
      a✝ : Not (Eq (g a) a)
      pb : Not (p (f.asEmbedding a))
      ⊢ False
    -/
    exact pb (Subtype.prop _)
    /-
      🎉 no goals
    -/


theorem card_support_extend_domain (f : α ≃ Subtype p) {g : Perm α} :
                                                   /-
                                                     α : Type u_1
                                                     inst✝⁴ : DecidableEq α
                                                     inst✝³ : Fintype α
                                                     β : Type u_2
                                                     inst✝² : DecidableEq β
                                                     inst✝¹ : Fintype β
                                                     p : β → Prop
                                                     inst✝ : DecidablePred p
                                                     f : Equiv α (Subtype p)
                                                     g : Equiv.Perm α
                                                     ⊢ Eq (g.extendDomain f).support.card g.support.card
                                                   -/
    #(g.extendDomain f).support = #g.support := by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem card_support_eq_zero {f : Perm α} : #f.support = 0 ↔ f = 1 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    ⊢ Iff (Eq f.support.card 0) (Eq f 1)
  -/
  rw [Finset.card_eq_zero, support_eq_empty_iff]
  /-
    🎉 no goals
  -/


theorem one_lt_card_support_of_ne_one {f : Perm α} (h : f ≠ 1) : 1 < #f.support := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    h : Ne f 1
    ⊢ LT.lt 1 f.support.card
  -/
  simp_rw [one_lt_card_iff, mem_support, ← not_or]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    h : Ne f 1
    ⊢ Exists fun a => Exists fun b => Not (Or (Eq (f a) a) (Or (Eq (f b) b) (Eq a  …
  -/
  contrapose! h
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    h : ∀ (a b : α), Or (Eq (f a) a) (Or (Eq (f b) b) (Eq a b))
    ⊢ Eq f 1
  -/
  ext a
  /-
    case H
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    h : ∀ (a b : α), Or (Eq (f a) a) (Or (Eq (f b) b) (Eq a b))
    a : α
    ⊢ Eq (f a) (1 a)
  -/
  specialize h (f a) a
  /-
    case H
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    a : α
    h : Or (Eq (f (f a)) (f a)) (Or (Eq (f a) a) (Eq (f a) a))
    ⊢ Eq (f a) (1 a)
  -/
  rwa [apply_eq_iff_eq, or_self_iff, or_self_iff] at h
  /-
    🎉 no goals
  -/


theorem card_support_ne_one (f : Perm α) : #f.support ≠ 1 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    ⊢ Ne f.support.card 1
  -/
  by_cases h : f = 1
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : Eq f 1
      ⊢ Ne f.support.card 1
    -/
  · exact ne_of_eq_of_ne (card_support_eq_zero.mpr h) zero_ne_one
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : Not (Eq f 1)
      ⊢ Ne f.support.card 1
    -/
  · exact ne_of_gt (one_lt_card_support_of_ne_one h)
    /-
      🎉 no goals
    -/


@[simp]
theorem card_support_le_one {f : Perm α} : #f.support ≤ 1 ↔ f = 1 := by
  rw [le_iff_lt_or_eq, Nat.lt_succ_iff, Nat.le_zero, card_support_eq_zero, or_iff_not_imp_right,
    imp_iff_right f.card_support_ne_one]


theorem two_le_card_support_of_ne_one {f : Perm α} (h : f ≠ 1) : 2 ≤ #f.support :=
  one_lt_card_support_of_ne_one h


theorem card_support_swap_mul {f : Perm α} {x : α} (hx : f x ≠ x) :
    #(swap x (f x) * f).support < #f.support :=
  Finset.card_lt_card
    ⟨fun _ hz => (mem_support_swap_mul_imp_mem_support_ne hz).left, fun h =>
                                                          /-
                                                            α : Type u_1
                                                            inst✝¹ : DecidableEq α
                                                            inst✝ : Fintype α
                                                            f : Equiv.Perm α
                                                            x : α
                                                            hx : Ne (f x) x
                                                            h : HasSubset.Subset f.support (HMul.hMul (Equiv.swap x (f x)) f).support
                                                            ⊢ Not (Ne ((HMul.hMul (Equiv.swap x (f x)) f) x) x)
                                                          -/
      absurd (h (mem_support.2 hx)) (mt mem_support.1 (by simp))⟩
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem card_support_swap {x y : α} (hxy : x ≠ y) : #(swap x y).support = 2 :=
                                                 /-
                                                   α : Type u_1
                                                   inst✝¹ : DecidableEq α
                                                   inst✝ : Fintype α
                                                   x y : α
                                                   hxy : Ne x y
                                                   ⊢ (Multiset.cons x (Multiset.cons y 0)).Nodup
                                                 -/
  show #(swap x y).support = #⟨x ::ₘ y ::ₘ 0, by simp [hxy]⟩ from
                                                 /-
                                                   🎉 no goals
                                                 -/
                         /-
                           α : Type u_1
                           inst✝¹ : DecidableEq α
                           inst✝ : Fintype α
                           x y : α
                           hxy : Ne x y
                           ⊢ Eq (Equiv.swap x y).support { val := Multiset.cons x (Multiset.cons y 0), no …
                         -/
    congr_arg card <| by simp [support_swap hxy, *, Finset.ext_iff]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem card_support_eq_two {f : Perm α} : #f.support = 2 ↔ IsSwap f := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f : Equiv.Perm α
    ⊢ Iff (Eq f.support.card 2) f.IsSwap
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : Eq f.support.card 2
      ⊢ f.IsSwap
    -/
  · obtain ⟨x, t, hmem, hins, ht⟩ := card_eq_succ.1 h
    /-
      case mp.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : Eq f.support.card 2
      x : α
      t : Finset α
      hmem : Not (Membership.mem t x)
      hins : Eq (Insert.insert x t) f.support
      ht : Eq t.card 1
      ⊢ f.IsSwap
    -/
    obtain ⟨y, rfl⟩ := card_eq_one.1 ht
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : Eq f.support.card 2
      x y : α
      hmem : Not (Membership.mem (Singleton.singleton y) x)
      hins : Eq (Insert.insert x (Singleton.singleton y)) f.support
      ht : Eq (Singleton.singleton y).card 1
      ⊢ f.IsSwap
    -/
    rw [mem_singleton] at hmem
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : Eq f.support.card 2
      x y : α
      hmem : Not (Eq x y)
      hins : Eq (Insert.insert x (Singleton.singleton y)) f.support
      ht : Eq (Singleton.singleton y).card 1
      ⊢ f.IsSwap
    -/
    refine ⟨x, y, hmem, ?_⟩
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : Eq f.support.card 2
      x y : α
      hmem : Not (Eq x y)
      hins : Eq (Insert.insert x (Singleton.singleton y)) f.support
      ht : Eq (Singleton.singleton y).card 1
      ⊢ Eq f (Equiv.swap x y)
    -/
    ext a
    /-
      case mp.intro.intro.intro.intro.intro.H
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : Eq f.support.card 2
      x y : α
      hmem : Not (Eq x y)
      hins : Eq (Insert.insert x (Singleton.singleton y)) f.support
      ht : Eq (Singleton.singleton y).card 1
      a : α
      ⊢ Eq (f a) ((Equiv.swap x y) a)
    -/
    have key : ∀ b, f b ≠ b ↔ _ := fun b => by rw [← mem_support, ← hins, mem_insert, mem_singleton]
    /-
      case mp.intro.intro.intro.intro.intro.H
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : Eq f.support.card 2
      x y : α
      hmem : Not (Eq x y)
      hins : Eq (Insert.insert x (Singleton.singleton y)) f.support
      ht : Eq (Singleton.singleton y).card 1
      a : α
      key : ∀ (b : α), Iff (Ne (f b) b) (Or (Eq b x) (Eq b y))
      ⊢ Eq (f a) ((Equiv.swap x y) a)
    -/
    by_cases ha : f a = a
      /-
        case pos
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f : Equiv.Perm α
        h : Eq f.support.card 2
        x y : α
        hmem : Not (Eq x y)
        hins : Eq (Insert.insert x (Singleton.singleton y)) f.support
        ht : Eq (Singleton.singleton y).card 1
        a : α
        key : ∀ (b : α), Iff (Ne (f b) b) (Or (Eq b x) (Eq b y))
        ha : Eq (f a) a
        ⊢ Eq (f a) ((Equiv.swap x y) a)
      -/
    · have ha' := not_or.mp (mt (key a).mpr (not_not.mpr ha))
      /-
        case pos
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f : Equiv.Perm α
        h : Eq f.support.card 2
        x y : α
        hmem : Not (Eq x y)
        hins : Eq (Insert.insert x (Singleton.singleton y)) f.support
        ht : Eq (Singleton.singleton y).card 1
        a : α
        key : ∀ (b : α), Iff (Ne (f b) b) (Or (Eq b x) (Eq b y))
        ha : Eq (f a) a
        ha' : And (Not (Eq a x)) (Not (Eq a y))
        ⊢ Eq (f a) ((Equiv.swap x y) a)
      -/
      rw [ha, swap_apply_of_ne_of_ne ha'.1 ha'.2]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f : Equiv.Perm α
        h : Eq f.support.card 2
        x y : α
        hmem : Not (Eq x y)
        hins : Eq (Insert.insert x (Singleton.singleton y)) f.support
        ht : Eq (Singleton.singleton y).card 1
        a : α
        key : ∀ (b : α), Iff (Ne (f b) b) (Or (Eq b x) (Eq b y))
        ha : Not (Eq (f a) a)
        ⊢ Eq (f a) ((Equiv.swap x y) a)
      -/
    · have ha' := (key (f a)).mp (mt f.apply_eq_iff_eq.mp ha)
      /-
        case neg
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        f : Equiv.Perm α
        h : Eq f.support.card 2
        x y : α
        hmem : Not (Eq x y)
        hins : Eq (Insert.insert x (Singleton.singleton y)) f.support
        ht : Eq (Singleton.singleton y).card 1
        a : α
        key : ∀ (b : α), Iff (Ne (f b) b) (Or (Eq b x) (Eq b y))
        ha : Not (Eq (f a) a)
        ha' : Or (Eq (f a) x) (Eq (f a) y)
        ⊢ Eq (f a) ((Equiv.swap x y) a)
      -/
      obtain rfl | rfl := (key a).mp ha
        /-
          case neg.inl
          α : Type u_1
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          f : Equiv.Perm α
          h : Eq f.support.card 2
          y : α
          ht : Eq (Singleton.singleton y).card 1
          a : α
          ha : Not (Eq (f a) a)
          hmem : Not (Eq a y)
          hins : Eq (Insert.insert a (Singleton.singleton y)) f.support
          key : ∀ (b : α), Iff (Ne (f b) b) (Or (Eq b a) (Eq b y))
          ha' : Or (Eq (f a) a) (Eq (f a) y)
          ⊢ Eq (f a) ((Equiv.swap a y) a)
        -/
      · rw [Or.resolve_left ha' ha, swap_apply_left]
        /-
          🎉 no goals
        -/
        /-
          case neg.inr
          α : Type u_1
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          f : Equiv.Perm α
          h : Eq f.support.card 2
          x a : α
          ha : Not (Eq (f a) a)
          hmem : Not (Eq x a)
          hins : Eq (Insert.insert x (Singleton.singleton a)) f.support
          ht : Eq (Singleton.singleton a).card 1
          key : ∀ (b : α), Iff (Ne (f b) b) (Or (Eq b x) (Eq b a))
          ha' : Or (Eq (f a) x) (Eq (f a) a)
          ⊢ Eq (f a) ((Equiv.swap x a) a)
        -/
      · rw [Or.resolve_right ha' ha, swap_apply_right]
        /-
          🎉 no goals
        -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f : Equiv.Perm α
      h : f.IsSwap
      ⊢ Eq f.support.card 2
    -/
  · obtain ⟨x, y, hxy, rfl⟩ := h
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      x y : α
      hxy : Ne x y
      ⊢ Eq (Equiv.swap x y).support.card 2
    -/
    exact card_support_swap hxy
    /-
      🎉 no goals
    -/


theorem Disjoint.card_support_mul (h : Disjoint f g) :
    #(f * g).support = #f.support + #g.support := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    f g : Equiv.Perm α
    h : f.Disjoint g
    ⊢ Eq (HMul.hMul f g).support.card (HAdd.hAdd f.support.card g.support.card)
  -/
  rw [← Finset.card_union_of_disjoint]
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : f.Disjoint g
      ⊢ Eq (HMul.hMul f g).support.card (Union.union f.support g.support).card
    -/
  · congr
    /-
      case e_s
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : f.Disjoint g
      ⊢ Eq (HMul.hMul f g).support (Union.union f.support g.support)
    -/
    ext
    /-
      case e_s.h
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : f.Disjoint g
      a✝ : α
      ⊢ Iff (Membership.mem (HMul.hMul f g).support a✝) (Membership.mem (Union.union …
    -/
    simp [h.support_mul]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      f g : Equiv.Perm α
      h : f.Disjoint g
      ⊢ _root_.Disjoint f.support g.support
    -/
  · simpa using h.disjoint_support
    /-
      🎉 no goals
    -/


theorem card_support_prod_list_of_pairwise_disjoint {l : List (Perm α)} (h : l.Pairwise Disjoint) :
    #l.prod.support = (l.map (card ∘ support)).sum := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    l : List (Equiv.Perm α)
    h : List.Pairwise Equiv.Perm.Disjoint l
    ⊢ Eq l.prod.support.card (List.map (Function.comp Finset.card Equiv.Perm.suppo …
  -/
  induction' l with a t ih
    /-
      case nil
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      h : List.Pairwise Equiv.Perm.Disjoint List.nil
      ⊢ Eq List.nil.prod.support.card (List.map (Function.comp Finset.card Equiv.Per …
    -/
  · exact card_support_eq_zero.mpr rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      a : Equiv.Perm α
      t : List (Equiv.Perm α)
      ih : List.Pairwise Equiv.Perm.Disjoint t → Eq t.prod.support.card (List.map (F …
      h : List.Pairwise Equiv.Perm.Disjoint (List.cons a t)
      ⊢ Eq (List.cons a t).prod.support.card (List.map (Function.comp Finset.card Eq …
    -/
  · obtain ⟨ha, ht⟩ := List.pairwise_cons.1 h
    /-
      case cons.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      a : Equiv.Perm α
      t : List (Equiv.Perm α)
      ih : List.Pairwise Equiv.Perm.Disjoint t → Eq t.prod.support.card (List.map (F …
      h : List.Pairwise Equiv.Perm.Disjoint (List.cons a t)
      ha : ∀ (a' : Equiv.Perm α), Membership.mem t a' → a.Disjoint a'
      ht : List.Pairwise Equiv.Perm.Disjoint t
      ⊢ Eq (List.cons a t).prod.support.card (List.map (Function.comp Finset.card Eq …
    -/
    rw [List.prod_cons, List.map_cons, List.sum_cons, ← ih ht]
    /-
      case cons.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      a : Equiv.Perm α
      t : List (Equiv.Perm α)
      ih : List.Pairwise Equiv.Perm.Disjoint t → Eq t.prod.support.card (List.map (F …
      h : List.Pairwise Equiv.Perm.Disjoint (List.cons a t)
      ha : ∀ (a' : Equiv.Perm α), Membership.mem t a' → a.Disjoint a'
      ht : List.Pairwise Equiv.Perm.Disjoint t
      ⊢ Eq (HMul.hMul a t.prod).support.card (HAdd.hAdd (Function.comp Finset.card E …
    -/
    exact (disjoint_prod_right _ ha).card_support_mul
    /-
      🎉 no goals
    -/


@[simp]
theorem support_subtype_perm [DecidableEq α] {s : Finset α} (f : Perm α) (h) :
    (f.subtypePerm h : Perm s).support = ({x | f x ≠ x} : Finset s) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    f : Equiv.Perm α
    h : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem s (f x))
    ⊢ Eq (f.subtypePerm h).support (Finset.filter (fun x => Ne (f ↑x) ↑x) Finset.u …
  -/
  ext; simp [Subtype.ext_iff]
       /-
         🎉 no goals
       -/


theorem fixed_point_card_lt_of_ne_one [DecidableEq α] [Fintype α] {σ : Perm α} (h : σ ≠ 1) :
    #{x | σ x = x} < Fintype.card α - 1 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    h : Ne σ 1
    ⊢ LT.lt (Finset.filter (fun x => Eq (σ x) x) Finset.univ).card (HSub.hSub (Fin …
  -/
  rw [Nat.lt_sub_iff_add_lt, ← Nat.lt_sub_iff_add_lt', ← Finset.card_compl, Finset.compl_filter]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    h : Ne σ 1
    ⊢ LT.lt 1 (Finset.filter (fun x => Not (Eq (σ x) x)) Finset.univ).card
  -/
  exact one_lt_card_support_of_ne_one h
  /-
    🎉 no goals
  -/


@[simp]
theorem support_conj : (σ * τ * σ⁻¹).support = τ.support.map σ.toEmbedding := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Equiv.Perm α
    ⊢ Eq (HMul.hMul (HMul.hMul σ τ) (Inv.inv σ)).support (Finset.map (Equiv.toEmbe …
  -/
  ext
  simp only [mem_map_equiv, Perm.coe_mul, Function.comp_apply, Ne, Perm.mem_support,
    Equiv.eq_symm_apply, inv_def]


                                                                      /-
                                                                        α : Type u_1
                                                                        inst✝¹ : Fintype α
                                                                        inst✝ : DecidableEq α
                                                                        σ τ : Equiv.Perm α
                                                                        ⊢ Eq (HMul.hMul (HMul.hMul σ τ) (Inv.inv σ)).support.card τ.support.card
                                                                      -/
theorem card_support_conj : #(σ * τ * σ⁻¹).support = #τ.support := by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


