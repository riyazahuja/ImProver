instance [∀ b, EDist (π b)] : EDist (∀ b, π b) where
  edist f g := Finset.sup univ fun b => edist (f b) (g b)


theorem edist_pi_def [∀ b, EDist (π b)] (f g : ∀ b, π b) :
    edist f g = Finset.sup univ fun b => edist (f b) (g b) :=
  rfl


theorem edist_le_pi_edist [∀ b, EDist (π b)] (f g : ∀ b, π b) (b : β) :
    edist (f b) (g b) ≤ edist f g :=
  le_sup (f := fun b => edist (f b) (g b)) (Finset.mem_univ b)


theorem edist_pi_le_iff [∀ b, EDist (π b)] {f g : ∀ b, π b} {d : ℝ≥0∞} :
    edist f g ≤ d ↔ ∀ b, edist (f b) (g b) ≤ d :=
                                /-
                                  β : Type v
                                  π : β → Type u_2
                                  inst✝¹ : Fintype β
                                  inst✝ : (b : β) → EDist (π b)
                                  f g : (b : β) → π b
                                  d : ENNReal
                                  ⊢ Iff (∀ (b : β), Membership.mem Finset.univ b → LE.le (EDist.edist (f b) (g b …
                                -/
  Finset.sup_le_iff.trans <| by simp only [Finset.mem_univ, forall_const]
                                /-
                                  🎉 no goals
                                -/


theorem edist_pi_const_le (a b : α) : (edist (fun _ : β => a) fun _ => b) ≤ edist a b :=
  edist_pi_le_iff.2 fun _ => le_rfl


@[simp]
theorem edist_pi_const [Nonempty β] (a b : α) : (edist (fun _ : β => a) fun _ => b) = edist a b :=
  Finset.sup_const univ_nonempty (edist a b)


/-- The product of a finite number of pseudoemetric spaces, with the max distance, is still
a pseudoemetric space.
This construction would also work for infinite products, but it would not give rise
to the product topology. Hence, we only formalize it in the good situation of finitely many
spaces. -/
instance pseudoEMetricSpacePi [∀ b, PseudoEMetricSpace (π b)] : PseudoEMetricSpace (∀ b, π b) where
                                                    /-
                                                      α : Type u
                                                      β : Type v
                                                      X : Type u_1
                                                      inst✝² : PseudoEMetricSpace α
                                                      π : β → Type u_2
                                                      inst✝¹ : Fintype β
                                                      inst✝ : (b : β) → PseudoEMetricSpace (π b)
                                                      f : (b : β) → π b
                                                      ⊢ ∀ (b : β), Membership.mem Finset.univ b → LE.le (EDist.edist (f b) (f b)) Bo …
                                                    -/
  edist_self f := bot_unique <| Finset.sup_le <| by simp
                                                    /-
                                                      🎉 no goals
                                                    -/
                       /-
                         α : Type u
                         β : Type v
                         X : Type u_1
                         inst✝² : PseudoEMetricSpace α
                         π : β → Type u_2
                         inst✝¹ : Fintype β
                         inst✝ : (b : β) → PseudoEMetricSpace (π b)
                         f g : (b : β) → π b
                         ⊢ Eq (EDist.edist f g) (EDist.edist g f)
                       -/
  edist_comm f g := by simp [edist_pi_def, edist_comm]
                       /-
                         🎉 no goals
                       -/
  edist_triangle _ g _ := edist_pi_le_iff.2 fun b => le_trans (edist_triangle _ (g b) _)
    (add_le_add (edist_le_pi_edist _ _ _) (edist_le_pi_edist _ _ _))
  toUniformSpace := Pi.uniformSpace _
  uniformity_edist := by
    simp only [Pi.uniformity, PseudoEMetricSpace.uniformity_edist, comap_iInf, gt_iff_lt,
      preimage_setOf_eq, comap_principal, edist_pi_def]
    /-
      α : Type u
      β : Type v
      X : Type u_1
      inst✝² : PseudoEMetricSpace α
      π : β → Type u_2
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoEMetricSpace (π b)
      ⊢ Eq (iInf fun i => iInf fun i_1 => iInf fun x => Filter.principal (setOf fun  …
    -/
    rw [iInf_comm]; congr; funext ε
    /-
      case e_s.h
      α : Type u
      β : Type v
      X : Type u_1
      inst✝² : PseudoEMetricSpace α
      π : β → Type u_2
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoEMetricSpace (π b)
      ε : ENNReal
      ⊢ Eq (iInf fun i => iInf fun x => Filter.principal (setOf fun a => LT.lt (EDis …
    -/
    rw [iInf_comm]; congr; funext εpos
    /-
      case e_s.h.e_s.h
      α : Type u
      β : Type v
      X : Type u_1
      inst✝² : PseudoEMetricSpace α
      π : β → Type u_2
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoEMetricSpace (π b)
      ε : ENNReal
      εpos : LT.lt 0 ε
      ⊢ Eq (iInf fun i => Filter.principal (setOf fun a => LT.lt (EDist.edist (a.1 i …
    -/
    simp [setOf_forall, εpos]
    /-
      🎉 no goals
    -/


/-- The product of a finite number of emetric spaces, with the max distance, is still
an emetric space.
This construction would also work for infinite products, but it would not give rise
to the product topology. Hence, we only formalize it in the good situation of finitely many
spaces. -/
instance emetricSpacePi [∀ b, EMetricSpace (π b)] : EMetricSpace (∀ b, π b) :=
  .ofT0PseudoEMetricSpace _


