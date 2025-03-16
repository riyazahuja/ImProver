theorem isThin_iff : Quiver.IsThin C ↔ ∀ c : C, Subsingleton (c ⟶ c) := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Groupoid C
    ⊢ Iff (Quiver.IsThin C) (∀ (c : C), Subsingleton (Quiver.Hom c c))
  -/
  refine ⟨fun h c => h c c, fun h c d => Subsingleton.intro fun f g => ?_⟩
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Groupoid C
    h : ∀ (c : C), Subsingleton (Quiver.Hom c c)
    c d : C
    f g : Quiver.Hom c d
    ⊢ Eq f g
  -/
  haveI := h d
  calc
    f = f ≫ inv g ≫ g := by simp only [inv_eq_inv, IsIso.inv_hom_id, Category.comp_id]
    _ = f ≫ inv f ≫ g := by congr 1
                            simp only [inv_eq_inv, IsIso.inv_hom_id, eq_iff_true_of_subsingleton]
    _ = g := by simp only [inv_eq_inv, IsIso.hom_inv_id_assoc]


/-- A subgroupoid is totally disconnected if it only has loops. -/
def IsTotallyDisconnected :=
  ∀ c d : C, (c ⟶ d) → c = d


