instance instNormalCommutatorClosure : (commutator G).topologicalClosure.Normal :=
  Subgroup.is_normal_topologicalClosure (commutator G)


/-- The topological abelianization of `absoluteGaloisGroup`, that is, the quotient of
  `absoluteGaloisGroup` by the topological closure of its commutator subgroup. -/
abbrev TopologicalAbelianization := G ⧸ Subgroup.topologicalClosure (commutator G)


local notation "G_ab" => TopologicalAbelianization


instance commGroup : CommGroup (G_ab G) where
  mul_comm := fun x y =>
    Quotient.inductionOn₂' x y fun a b =>
      Quotient.sound' <|
        QuotientGroup.leftRel_apply.mpr <| by
          /-
            G : Type u_1
            inst✝² : Group G
            inst✝¹ : TopologicalSpace G
            inst✝ : TopologicalGroup G
            x y : TopologicalAbelianization G
            a b : G
            ⊢ Membership.mem (commutator G).topologicalClosure (HMul.hMul (Inv.inv ((fun x …
          -/
          have h : (a * b)⁻¹ * (b * a) = ⁅b⁻¹, a⁻¹⁆ := by group
          /-
            G : Type u_1
            inst✝² : Group G
            inst✝¹ : TopologicalSpace G
            inst✝ : TopologicalGroup G
            x y : TopologicalAbelianization G
            a b : G
            h : Eq (HMul.hMul (Inv.inv (HMul.hMul a b)) (HMul.hMul b a)) (Bracket.bracket  …
            ⊢ Membership.mem (commutator G).topologicalClosure (HMul.hMul (Inv.inv ((fun x …
          -/
          rw [h]
          exact Subgroup.le_topologicalClosure _ (Subgroup.commutator_mem_commutator
            (Subgroup.mem_top b⁻¹) (Subgroup.mem_top a⁻¹))
  __ : Group (G_ab G) := inferInstance


