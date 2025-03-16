/-- Over `Fin (n + 1)`, the multisets of size `k + 1` containing `0` are equivalent to those of size
`k`, as demonstrated by respectively erasing or appending `0`. -/
protected def e1 {n k : ℕ} : { s : Sym (Fin (n + 1)) (k + 1) // ↑0 ∈ s } ≃ Sym (Fin n.succ) k where
  toFun s := s.1.erase 0 s.2
  invFun s := ⟨cons 0 s, mem_cons_self 0 s⟩
                   /-
                     α : Type u_1
                     n✝ n k : Nat
                     s : Subtype fun s => Membership.mem s 0
                     ⊢ Eq ((fun s => ⟨Sym.cons 0 s, ⋯⟩) ((fun s => (↑s).erase 0 ⋯) s)) s
                   -/
  left_inv s := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      α : Type u_1
                      n✝ n k : Nat
                      s : Sym (Fin n.succ) k
                      ⊢ Eq ((fun s => (↑s).erase 0 ⋯) ((fun s => ⟨Sym.cons 0 s, ⋯⟩) s)) s
                    -/
  right_inv s := by simp
                    /-
                      🎉 no goals
                    -/


/-- The multisets of size `k` over `Fin n+2` not containing `0`
are equivalent to those of size `k` over `Fin n+1`,
as demonstrated by respectively decrementing or incrementing every element of the multiset.
-/
protected def e2 {n k : ℕ} : { s : Sym (Fin n.succ.succ) k // ↑0 ∉ s } ≃ Sym (Fin n.succ) k where
  toFun s := map (Fin.predAbove 0) s.1
  invFun s :=
    ⟨map (Fin.succAbove 0) s,
      (mt mem_map.1) (not_exists.2 fun t => not_and.2 fun _ => Fin.succAbove_ne _ t)⟩
  left_inv s := by
    /-
      α : Type u_1
      n✝ n k : Nat
      s : Subtype fun s => Not (Membership.mem s 0)
      ⊢ Eq ((fun s => ⟨Sym.map (Fin.succAbove 0) s, ⋯⟩) ((fun s => Sym.map (Fin.pred …
    -/
    ext1
    /-
      case a
      α : Type u_1
      n✝ n k : Nat
      s : Subtype fun s => Not (Membership.mem s 0)
      ⊢ Eq ↑((fun s => ⟨Sym.map (Fin.succAbove 0) s, ⋯⟩) ((fun s => Sym.map (Fin.pre …
    -/
    simp only [map_map]
    /-
      case a
      α : Type u_1
      n✝ n k : Nat
      s : Subtype fun s => Not (Membership.mem s 0)
      ⊢ Eq (Sym.map (Function.comp (Fin.succAbove 0) (Fin.predAbove 0)) ↑s) ↑s
    -/
    refine (Sym.map_congr fun v hv ↦ ?_).trans (map_id' _)
    /-
      case a
      α : Type u_1
      n✝ n k : Nat
      s : Subtype fun s => Not (Membership.mem s 0)
      v : Fin (HAdd.hAdd n.succ 1)
      hv : Membership.mem (↑s) v
      ⊢ Eq (Function.comp (Fin.succAbove 0) (Fin.predAbove 0) v) v
    -/
    exact Fin.succAbove_predAbove (ne_of_mem_of_not_mem hv s.2)
    /-
      🎉 no goals
    -/
  right_inv s := by
    /-
      α : Type u_1
      n✝ n k : Nat
      s : Sym (Fin n.succ) k
      ⊢ Eq ((fun s => Sym.map (Fin.predAbove 0) ↑s) ((fun s => ⟨Sym.map (Fin.succAbo …
    -/
    simp only [map_map, comp_apply, ← Fin.castSucc_zero, Fin.predAbove_succAbove, map_id']
    /-
      🎉 no goals
    -/

-- Porting note: use eqn compiler instead of `pincerRecursion` to make cases more readable

theorem card_sym_fin_eq_multichoose : ∀ n k : ℕ, card (Sym (Fin n) k) = multichoose n k
               /-
                 n : Nat
                 ⊢ Eq (Fintype.card (Sym (Fin n) 0)) (n.multichoose 0)
               -/
  | n, 0 => by simp
               /-
                 🎉 no goals
               -/
                   /-
                     k : Nat
                     ⊢ Eq (Fintype.card (Sym (Fin 0) (HAdd.hAdd k 1))) (Nat.multichoose 0 (HAdd.hAd …
                   -/
  | 0, k + 1 => by rw [multichoose_zero_succ]; exact card_eq_zero
                                               /-
                                                 🎉 no goals
                                               -/
                   /-
                     k : Nat
                     ⊢ Eq (Fintype.card (Sym (Fin 1) (HAdd.hAdd k 1))) (Nat.multichoose 1 (HAdd.hAd …
                   -/
  | 1, k + 1 => by simp
                   /-
                     🎉 no goals
                   -/
  | n + 2, k + 1 => by
    rw [multichoose_succ_succ, ← card_sym_fin_eq_multichoose (n + 1) (k + 1),
      ← card_sym_fin_eq_multichoose (n + 2) k, add_comm (Fintype.card _), ← card_sum]
    /-
      n k : Nat
      ⊢ Eq (Fintype.card (Sym (Fin (HAdd.hAdd n 2)) (HAdd.hAdd k 1))) (Fintype.card  …
    -/
    refine Fintype.card_congr (Equiv.symm ?_)
    /-
      n k : Nat
      ⊢ Equiv (Sum (Sym (Fin (HAdd.hAdd n 2)) k) (Sym (Fin (HAdd.hAdd n 1)) (HAdd.hA …
    -/
    apply (Sym.e1.symm.sumCongr Sym.e2.symm).trans
    /-
      n k : Nat
      ⊢ Equiv (Sum (Subtype fun s => Membership.mem s 0) (Subtype fun s => Not (Memb …
    -/
    apply Equiv.sumCompl
    /-
      🎉 no goals
    -/


/-- For any fintype `α` of cardinality `n`, `card (Sym α k) = multichoose (card α) k`. -/
theorem card_sym_eq_multichoose (α : Type*) (k : ℕ) [Fintype α] [Fintype (Sym α k)] :
    card (Sym α k) = multichoose (card α) k := by
  /-
    α : Type u_2
    k : Nat
    inst✝¹ : Fintype α
    inst✝ : Fintype (Sym α k)
    ⊢ Eq (Fintype.card (Sym α k)) ((Fintype.card α).multichoose k)
  -/
  rw [← card_sym_fin_eq_multichoose]
  -- FIXME: Without the `Fintype` namespace, why does it complain about `Finset.card_congr` being
  -- deprecated?
  /-
    α : Type u_2
    k : Nat
    inst✝¹ : Fintype α
    inst✝ : Fintype (Sym α k)
    ⊢ Eq (Fintype.card (Sym α k)) (Fintype.card (Sym (Fin (Fintype.card α)) k))
  -/
  exact Fintype.card_congr (equivCongr (equivFin α))
  /-
    🎉 no goals
  -/


/-- The *stars and bars* lemma: the cardinality of `Sym α k` is equal to
`Nat.choose (card α + k - 1) k`. -/
theorem card_sym_eq_choose {α : Type*} [Fintype α] (k : ℕ) [Fintype (Sym α k)] :
    card (Sym α k) = (card α + k - 1).choose k := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    k : Nat
    inst✝ : Fintype (Sym α k)
    ⊢ Eq (Fintype.card (Sym α k)) ((HSub.hSub (HAdd.hAdd (Fintype.card α) k) 1).ch …
  -/
  rw [card_sym_eq_multichoose, Nat.multichoose_eq]
  /-
    🎉 no goals
  -/


/-- The `diag` of `s : Finset α` is sent on a finset of `Sym2 α` of card `#s`. -/
theorem card_image_diag (s : Finset α) : #(s.diag.image Sym2.mk) = #s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (Finset.image Sym2.mk s.diag).card s.card
  -/
  rw [card_image_of_injOn, diag_card]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Set.InjOn Sym2.mk ↑s.diag
  -/
  rintro ⟨x₀, x₁⟩ hx _ _ h
  /-
    case mk
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x₀ x₁ : α
    hx : Membership.mem ↑s.diag { fst := x₀, snd := x₁ }
    x₂✝ : Prod α α
    a✝ : Membership.mem (↑s.diag) x₂✝
    h : Eq (Sym2.mk { fst := x₀, snd := x₁ }) (Sym2.mk x₂✝)
    ⊢ Eq { fst := x₀, snd := x₁ } x₂✝
  -/
  cases Sym2.eq.1 h
    /-
      case mk.refl
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x₀ x₁ : α
      hx a✝ : Membership.mem ↑s.diag { fst := x₀, snd := x₁ }
      h : Eq (Sym2.mk { fst := x₀, snd := x₁ }) (Sym2.mk { fst := x₀, snd := x₁ })
      ⊢ Eq { fst := x₀, snd := x₁ } { fst := x₀, snd := x₁ }
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case mk.swap
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x₀ x₁ : α
      hx : Membership.mem ↑s.diag { fst := x₀, snd := x₁ }
      a✝ : Membership.mem ↑s.diag { fst := x₁, snd := x₀ }
      h : Eq (Sym2.mk { fst := x₀, snd := x₁ }) (Sym2.mk { fst := x₁, snd := x₀ })
      ⊢ Eq { fst := x₀, snd := x₁ } { fst := x₁, snd := x₀ }
    -/
  · simp only [mem_coe, mem_diag] at hx
    /-
      case mk.swap
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x₀ x₁ : α
      a✝ : Membership.mem ↑s.diag { fst := x₁, snd := x₀ }
      h : Eq (Sym2.mk { fst := x₀, snd := x₁ }) (Sym2.mk { fst := x₁, snd := x₀ })
      hx : And (Membership.mem s x₀) (Eq x₀ x₁)
      ⊢ Eq { fst := x₀, snd := x₁ } { fst := x₁, snd := x₀ }
    -/
    rw [hx.2]
    /-
      🎉 no goals
    -/


lemma two_mul_card_image_offDiag (s : Finset α) : 2 * #(s.offDiag.image Sym2.mk) = #s.offDiag := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (HMul.hMul 2 (Finset.image Sym2.mk s.offDiag).card) s.offDiag.card
  -/
  rw [card_eq_sum_card_image (Sym2.mk : α × α → _), sum_const_nat (Sym2.ind _), mul_comm]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ ∀ (x y : α), Membership.mem (Finset.image Sym2.mk s.offDiag) (Sym2.mk { fst  …
  -/
  rintro x y hxy
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x y : α
    hxy : Membership.mem (Finset.image Sym2.mk s.offDiag) (Sym2.mk { fst := x, snd …
    ⊢ Eq (Finset.filter (fun a => Eq (Sym2.mk a) (Sym2.mk { fst := x, snd := y })) …
  -/
  simp_rw [mem_image, mem_offDiag] at hxy
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x y : α
    hxy : Exists fun a => And (And (Membership.mem s a.1) (And (Membership.mem s a …
    ⊢ Eq (Finset.filter (fun a => Eq (Sym2.mk a) (Sym2.mk { fst := x, snd := y })) …
  -/
  obtain ⟨a, ⟨ha₁, ha₂, ha⟩, h⟩ := hxy
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x y : α
    a : Prod α α
    h : Eq (Sym2.mk a) (Sym2.mk { fst := x, snd := y })
    ha₁ : Membership.mem s a.1
    ha₂ : Membership.mem s a.2
    ha : Ne a.1 a.2
    ⊢ Eq (Finset.filter (fun a => Eq (Sym2.mk a) (Sym2.mk { fst := x, snd := y })) …
  -/
  replace h := Sym2.eq.1 h
  obtain ⟨hx, hy, hxy⟩ : x ∈ s ∧ y ∈ s ∧ x ≠ y := by
    cases h <;> refine ⟨‹_›, ‹_›, ?_⟩ <;> [exact ha; exact ha.symm]
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x y : α
    a : Prod α α
    ha₁ : Membership.mem s a.1
    ha₂ : Membership.mem s a.2
    ha : Ne a.1 a.2
    h : Sym2.Rel α a { fst := x, snd := y }
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : Ne x y
    ⊢ Eq (Finset.filter (fun a => Eq (Sym2.mk a) (Sym2.mk { fst := x, snd := y })) …
  -/
  have hxy' : y ≠ x := hxy.symm
  have : {z ∈ s.offDiag | Sym2.mk z = s(x, y)} = {(x, y), (y, x)} := by
    ext ⟨x₁, y₁⟩
    rw [mem_filter, mem_insert, mem_singleton, Sym2.eq_iff, Prod.mk.inj_iff, Prod.mk.inj_iff,
      and_iff_right_iff_imp]
    -- `hxy'` is used in `exact`
    rintro (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩) <;> rw [mem_offDiag] <;> exact ⟨‹_›, ‹_›, ‹_›⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x y : α
    a : Prod α α
    ha₁ : Membership.mem s a.1
    ha₂ : Membership.mem s a.2
    ha : Ne a.1 a.2
    h : Sym2.Rel α a { fst := x, snd := y }
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : Ne x y
    hxy' : Ne y x
    this : Eq (Finset.filter (fun z => Eq (Sym2.mk z) (Sym2.mk { fst := x, snd :=  …
    ⊢ Eq (Finset.filter (fun a => Eq (Sym2.mk a) (Sym2.mk { fst := x, snd := y })) …
  -/
  rw [this, card_insert_of_not_mem, card_singleton]
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x y : α
    a : Prod α α
    ha₁ : Membership.mem s a.1
    ha₂ : Membership.mem s a.2
    ha : Ne a.1 a.2
    h : Sym2.Rel α a { fst := x, snd := y }
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : Ne x y
    hxy' : Ne y x
    this : Eq (Finset.filter (fun z => Eq (Sym2.mk z) (Sym2.mk { fst := x, snd :=  …
    ⊢ Not (Membership.mem (Singleton.singleton { fst := y, snd := x }) { fst := x, …
  -/
  simp only [not_and, Prod.mk.inj_iff, mem_singleton]
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x y : α
    a : Prod α α
    ha₁ : Membership.mem s a.1
    ha₂ : Membership.mem s a.2
    ha : Ne a.1 a.2
    h : Sym2.Rel α a { fst := x, snd := y }
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : Ne x y
    hxy' : Ne y x
    this : Eq (Finset.filter (fun z => Eq (Sym2.mk z) (Sym2.mk { fst := x, snd :=  …
    ⊢ Eq x y → Not (Eq y x)
  -/
  exact fun _ => hxy'
  /-
    🎉 no goals
  -/


/-- The `offDiag` of `s : Finset α` is sent on a finset of `Sym2 α` of card `#s.offDiag / 2`.
This is because every element `s(x, y)` of `Sym2 α` not on the diagonal comes from exactly two
pairs: `(x, y)` and `(y, x)`. -/
theorem card_image_offDiag (s : Finset α) : #(s.offDiag.image Sym2.mk) = (#s).choose 2 := by
  rw [Nat.choose_two_right, Nat.mul_sub_left_distrib, mul_one, ← offDiag_card,
    Nat.div_eq_of_eq_mul_right Nat.zero_lt_two (two_mul_card_image_offDiag s).symm]


theorem card_subtype_diag [Fintype α] : card { a : Sym2 α // a.IsDiag } = card α := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Eq (Fintype.card (Subtype fun a => a.IsDiag)) (Fintype.card α)
  -/
  convert card_image_diag (univ : Finset α)
  /-
    case h.e'_2
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Eq (Fintype.card (Subtype fun a => a.IsDiag)) (Finset.image Sym2.mk Finset.u …
  -/
  rw [← filter_image_mk_isDiag, Fintype.card_of_subtype]
  /-
    case h.e'_2.H
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ ∀ (x : Sym2 α), Iff (Membership.mem (Finset.filter (fun a => a.IsDiag) (Fins …
  -/
  rintro x
  /-
    case h.e'_2.H
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x : Sym2 α
    ⊢ Iff (Membership.mem (Finset.filter (fun a => a.IsDiag) (Finset.image Sym2.mk …
  -/
  rw [mem_filter, univ_product_univ, mem_image]
  /-
    case h.e'_2.H
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x : Sym2 α
    ⊢ Iff (And (Exists fun a => And (Membership.mem Finset.univ a) (Eq (Sym2.mk a) …
  -/
  obtain ⟨a, ha⟩ := Quot.exists_rep x
  /-
    case h.e'_2.H.intro
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x : Sym2 α
    a : Prod α α
    ha : Eq (Quot.mk (Sym2.Rel α) a) x
    ⊢ Iff (And (Exists fun a => And (Membership.mem Finset.univ a) (Eq (Sym2.mk a) …
  -/
  exact and_iff_right ⟨a, mem_univ _, ha⟩
  /-
    🎉 no goals
  -/


theorem card_subtype_not_diag [Fintype α] :
    card { a : Sym2 α // ¬a.IsDiag } = (card α).choose 2 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Eq (Fintype.card (Subtype fun a => Not a.IsDiag)) ((Fintype.card α).choose 2)
  -/
  convert card_image_offDiag (univ : Finset α)
  /-
    case h.e'_2
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ Eq (Fintype.card (Subtype fun a => Not a.IsDiag)) (Finset.image Sym2.mk Fins …
  -/
  rw [← filter_image_mk_not_isDiag, Fintype.card_of_subtype]
  /-
    case h.e'_2.H
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    ⊢ ∀ (x : Sym2 α), Iff (Membership.mem (Finset.filter (fun a => Not a.IsDiag) ( …
  -/
  rintro x
  /-
    case h.e'_2.H
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x : Sym2 α
    ⊢ Iff (Membership.mem (Finset.filter (fun a => Not a.IsDiag) (Finset.image Sym …
  -/
  rw [mem_filter, univ_product_univ, mem_image]
  /-
    case h.e'_2.H
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x : Sym2 α
    ⊢ Iff (And (Exists fun a => And (Membership.mem Finset.univ a) (Eq (Sym2.mk a) …
  -/
  obtain ⟨a, ha⟩ := Quot.exists_rep x
  /-
    case h.e'_2.H.intro
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    x : Sym2 α
    a : Prod α α
    ha : Eq (Quot.mk (Sym2.Rel α) a) x
    ⊢ Iff (And (Exists fun a => And (Membership.mem Finset.univ a) (Eq (Sym2.mk a) …
  -/
  exact and_iff_right ⟨a, mem_univ _, ha⟩
  /-
    🎉 no goals
  -/


/-- Type **stars and bars** for the case `n = 2`. -/
protected theorem card {α} [Fintype α] : card (Sym2 α) = Nat.choose (card α + 1) 2 :=
  Finset.card_sym2 _


