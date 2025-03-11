/-- The supremum of the two elements. -/
def sup [SemilatticeSup α] (x : Sym2 α) : α := Sym2.lift ⟨(· ⊔ ·), sup_comm⟩ x


@[simp] theorem sup_mk [SemilatticeSup α] (a b : α) : s(a, b).sup = a ⊔ b := rfl


/-- The infimum of the two elements. -/
def inf [SemilatticeInf α] (x : Sym2 α) : α := Sym2.lift ⟨(· ⊓ ·), inf_comm⟩ x


@[simp] theorem inf_mk [SemilatticeInf α] (a b : α) : s(a, b).inf = a ⊓ b := rfl


protected theorem inf_le_sup [Lattice α] (s : Sym2 α) : s.inf ≤ s.sup := by
  /-
    α : Type u_1
    inst✝ : Lattice α
    s : Sym2 α
    ⊢ LE.le s.inf s.sup
  -/
  cases s using Sym2.ind; simp [_root_.inf_le_sup]
                          /-
                            🎉 no goals
                          -/


/-- In a linear order, symmetric squares are canonically identified with ordered pairs. -/
@[simps!]
def sortEquiv [LinearOrder α] : Sym2 α ≃ { p : α × α // p.1 ≤ p.2 } where
  toFun s := ⟨(s.inf, s.sup), Sym2.inf_le_sup _⟩
  invFun p := Sym2.mk p
  left_inv := Sym2.ind fun a b => mk_eq_mk_iff.mpr <| by
    cases le_total a b with
    | inl h => simp [h]
    | inr h => simp [h]
  right_inv := Subtype.rec <| Prod.rec fun x y hxy =>
                                /-
                                  α : Type ?u.2011
                                  inst✝ : LinearOrder α
                                  x y : α
                                  hxy : LE.le { fst := x, snd := y }.1 { fst := x, snd := y }.2
                                  ⊢ Eq (↑((fun s => ⟨{ fst := s.inf, snd := s.sup }, ⋯⟩) ((fun p => Sym2.mk ↑p)  …
                                -/
                                /-
                                  🎉 no goals
                                -/
    Subtype.ext <| Prod.ext (by simp [hxy]) (by simp [hxy])
                                                /-
                                                  🎉 no goals
                                                -/


